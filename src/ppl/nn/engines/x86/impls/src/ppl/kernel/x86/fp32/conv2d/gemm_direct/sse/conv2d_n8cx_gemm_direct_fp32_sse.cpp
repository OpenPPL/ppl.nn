// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <new>
#include <nmmintrin.h>
#include <string.h>

#include "ppl/kernel/x86/fp32/reorder.h"
#include "ppl/kernel/x86/common/sse_tools.h"
#include "ppl/kernel/x86/fp32/conv2d/gemm_direct/sse/conv2d_n8cx_gemm_direct_fp32_sse.h"
#include "ppl/kernel/x86/fp32/conv2d/gemm_direct/sse/conv2d_n8cx_gemm_direct_kernel_fp32_sse.h"
#include "ppl/common/sys.h"

#define ASSUME_L2_BYTES() (256 * 1024)
#define ASSUME_L2_WAYS()  4
#define ASSUME_L3_BYTES() (2048 * 1024)
#define L2_RATIO()        0.251
#define L3_RATIO()        0.501

#define IC_L2_BLK_MAX_L()      (12 * CH_DT_BLK()) // preserve for tuning
#define IC_L2_BLK_MAX_S()      (12 * CH_DT_BLK())
#define IC_L2_BLK_TAIL_RATIO() 0.251
#define OC_L2_BLK_MAX_L()      (12 * CH_DT_BLK()) // preserve for tuning
#define OC_L2_BLK_MAX_S()      (12 * CH_DT_BLK())
#define OC_L2_BLK_MIN()        (2 * CH_DT_BLK())
#define HW_L2_BLK_MAX()        256
#define HW_L2_BLK_MIN()        BLK1X3_HW_RF()

namespace ppl { namespace kernel { namespace x86 {

int64_t conv2d_n8cx_gemm_direct_fp32_sse_executor::cal_ic_l2_blk(const conv2d_fp32_param &param)
{
    const int64_t ic_per_gp = param.channels / param.group;
    const int64_t padded_ic = round_up(ic_per_gp, CH_DT_BLK());
    const int64_t oc_per_gp = param.num_output / param.group;
    const int64_t padded_oc = round_up(oc_per_gp, CH_DT_BLK());

    int64_t ic_l2_blk;
    if (padded_ic > padded_oc) {
        ic_l2_blk = min<int64_t>(IC_L2_BLK_MAX_L(), padded_ic);
    } else {
        ic_l2_blk = min<int64_t>(IC_L2_BLK_MAX_S(), padded_ic);
    }
    if (mod_up(padded_ic, ic_l2_blk) < IC_L2_BLK_TAIL_RATIO() * ic_l2_blk) {
        ic_l2_blk = round_up(padded_ic / (padded_ic / ic_l2_blk), CH_DT_BLK());
    }
    return ic_l2_blk;
}

void conv2d_n8cx_gemm_direct_fp32_sse_executor::init_preproc_param()
{
    schedule_param_.ic_per_gp = conv_param_->channels / conv_param_->group;
    schedule_param_.oc_per_gp = conv_param_->num_output / conv_param_->group;
    schedule_param_.padded_ic = round_up(schedule_param_.ic_per_gp, CH_DT_BLK());
    schedule_param_.padded_oc = round_up(schedule_param_.oc_per_gp, CH_DT_BLK());
}

void conv2d_n8cx_gemm_direct_fp32_sse_executor::cal_kernel_tunning_param()
{
    const conv2d_fp32_param &cp = *conv_param_;
    kernel_schedule_param &sp   = schedule_param_;

    const int64_t num_thread = PPL_OMP_MAX_THREADS();
    const int64_t batch      = src_shape_->GetDim(0);
    const int64_t dst_hw     = dst_shape_->GetDim(2) * dst_shape_->GetDim(3);

    const float l3_cap_all_core = (ppl::common::GetCpuCacheL3() == 0 ? (ASSUME_L3_BYTES() * num_thread) : ppl::common::GetCpuCacheL3()) * L3_RATIO() / sizeof(float);

    sp.ic_l2_blk = cal_ic_l2_blk(cp);
    sp.ic_l2_cnt = div_up(sp.padded_ic, sp.ic_l2_blk);

    sp.gp_l3_blk = min(cp.group, num_thread);
    sp.mb_l3_blk = min(batch, div_up(num_thread, sp.gp_l3_blk));
    while (sp.gp_l3_blk > 1 && sp.gp_l3_blk * sp.mb_l3_blk * sp.ic_l2_blk * dst_hw > l3_cap_all_core) {
        --sp.gp_l3_blk;
    }
    sp.mb_l3_blk = min(batch, div_up(num_thread, sp.gp_l3_blk));
    while (sp.mb_l3_blk > 1 && sp.gp_l3_blk * sp.mb_l3_blk * sp.ic_l2_blk * dst_hw > l3_cap_all_core) {
        --sp.mb_l3_blk;
    }

    sp.oc_kr_blk = min<int64_t>(BLK1X1_OC_RF() * CH_RF_BLK(), sp.padded_oc);
    if (sp.padded_oc % sp.oc_kr_blk != 0 && sp.padded_oc / sp.oc_kr_blk < 4) {
        sp.oc_kr_blk = BLK1X3_OC_RF() * CH_RF_BLK();
    }
    if (sp.padded_oc > sp.padded_ic) {
        sp.oc_l2_blk = min<int64_t>(OC_L2_BLK_MAX_L(), sp.padded_oc);
    } else {
        sp.oc_l2_blk = min<int64_t>(OC_L2_BLK_MAX_S(), sp.padded_oc);
    }

    static const int64_t hw_rf_table[6] = { 3, 3, 1, 1, 1, 1 };
    sp.hw_kr_blk = hw_rf_table[sp.oc_kr_blk / CH_DT_BLK() - 1];
    sp.hw_l2_blk = min<int64_t>(dst_hw, round_up(HW_L2_BLK_MAX(), sp.hw_kr_blk));

    if (sp.padded_oc > 2 * sp.oc_l2_blk && sp.padded_ic > IC_L2_BLK_MAX_L()) {
        const int64_t bghw_task = sp.gp_l3_blk * sp.mb_l3_blk * div_up(dst_hw, sp.hw_l2_blk);
        while (sp.oc_l2_blk - sp.oc_kr_blk >= OC_L2_BLK_MIN() && bghw_task * div_up(sp.padded_oc, sp.oc_l2_blk) < num_thread) {
            sp.oc_l2_blk -= sp.oc_kr_blk;
        }
    } else {
        const int64_t bgo_task = sp.gp_l3_blk * sp.mb_l3_blk * div_up(sp.padded_oc, sp.oc_l2_blk);
        while (sp.hw_l2_blk - sp.hw_kr_blk >= HW_L2_BLK_MIN() && bgo_task * div_up(dst_hw, sp.hw_l2_blk) < num_thread) {
            sp.hw_l2_blk -= sp.hw_kr_blk;
        }
        const int64_t bghw_task = sp.gp_l3_blk * sp.mb_l3_blk * div_up(dst_hw, sp.hw_l2_blk);
        while (sp.oc_l2_blk - sp.oc_kr_blk >= OC_L2_BLK_MIN() && bghw_task * div_up(sp.padded_oc, sp.oc_l2_blk) < num_thread) {
            sp.oc_l2_blk -= sp.oc_kr_blk;
        }
    }

    sp.down_sample = 0;
    if (cp.stride_h > 1 || cp.stride_w > 1) {
        sp.down_sample = 1;
    }

    sp.use_nt_store = 0;
    if (batch * cp.group * sp.padded_oc * dst_hw > l3_cap_all_core * 3) {
        sp.use_nt_store = 1;
    }
}

uint64_t conv2d_n8cx_gemm_direct_fp32_sse_executor::cal_temp_buffer_size()
{
    if (schedule_param_.down_sample) {
        const int64_t dst_hw = dst_shape_->GetDim(2) * dst_shape_->GetDim(3);
        return (uint64_t)dst_hw * schedule_param_.mb_l3_blk * schedule_param_.gp_l3_blk * schedule_param_.ic_l2_blk * sizeof(float);
    }
    return 64u;
}

ppl::common::RetCode conv2d_n8cx_gemm_direct_fp32_sse_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_ || ((conv_param_->fuse_flag & conv_fuse_flag::sum) && !sum_src_shape_)) {
        return ppl::common::RC_INVALID_VALUE;
    }

    init_preproc_param();
    cal_kernel_tunning_param();

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n8cx_gemm_direct_fp32_sse_executor::execute()
{
    if (!conv_param_ || !cvt_filter_ || !cvt_bias_ || !src_ || !dst_ || ((conv_param_->fuse_flag & conv_fuse_flag::sum) && !sum_src_) || !temp_buffer_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const conv2d_fp32_param &cp     = *conv_param_;
    const kernel_schedule_param &sp = schedule_param_;

    const int64_t batch = src_shape_->GetDim(0);
    const int64_t src_h = src_shape_->GetDim(2);
    const int64_t src_w = src_shape_->GetDim(3);
    const int64_t dst_hw = dst_shape_->GetDim(2) * dst_shape_->GetDim(3);

    const int64_t src_b_stride   = round_up(src_shape_->GetDim(1), CH_DT_BLK()) * src_h * src_w;
    const int64_t src_g_stride   = sp.padded_ic * src_h * src_w;
    const int64_t src_icb_stride = src_h * src_w * CH_DT_BLK();
    const int64_t src_h_stride   = src_w * CH_DT_BLK();
    const int64_t dst_b_stride   = round_up(dst_shape_->GetDim(1), CH_DT_BLK()) * dst_hw;
    const int64_t dst_g_stride   = sp.padded_oc * dst_hw;
    const int64_t dst_ocb_stride = dst_hw * CH_DT_BLK();
    const int64_t flt_g_stride   = sp.ic_l2_cnt * sp.padded_oc * sp.ic_l2_blk;
    const int64_t flt_ocb_stride = sp.ic_l2_blk * CH_DT_BLK();

    const bool with_sum   = cp.fuse_flag & conv_fuse_flag::sum;
    const bool with_relu  = cp.fuse_flag & conv_fuse_flag::relu;
    const bool with_relu6 = cp.fuse_flag & conv_fuse_flag::relu6;

    int64_t sum_src_b_stride = 0;
    if (with_sum) {
        sum_src_b_stride = int64_t(round_up(sum_src_shape_->GetDim(1), CH_DT_BLK())) * dst_hw;
    }

    int64_t share_param[SHAR_PARAM_LEN()];
    share_param[HIS_OCB_STRIDE_IDX()] = dst_ocb_stride;
    share_param[DST_OCB_STRIDE_IDX()] = dst_ocb_stride;
    share_param[FLT_OCB_STRIDE_IDX()] = flt_ocb_stride;
    const int64_t nt_store_sel = sp.use_nt_store;
    for (int64_t mbl3 = 0; mbl3 < batch; mbl3 += sp.mb_l3_blk) {
        const int64_t mbl3_eff = min<int64_t>(batch - mbl3, sp.mb_l3_blk);
        for (int64_t gpl3 = 0; gpl3 < cp.group; gpl3 += sp.gp_l3_blk) {
            const int64_t gpl3_eff = min<int64_t>(cp.group - gpl3, sp.gp_l3_blk);
            for (int64_t icl2 = 0; icl2 < sp.padded_ic; icl2 += sp.ic_l2_blk) {
                const int64_t icl2_eff = min<int64_t>(sp.ic_per_gp - icl2, sp.ic_l2_blk);
                const bool is_first_ic = icl2 == 0;
                const bool is_last_ic  = (icl2 + sp.ic_l2_blk >= sp.ic_per_gp);
                const float *base_src = src_;
                const float *base_his = dst_;
                const float *base_flt = cvt_filter_;
                float *base_dst       = dst_;

                int64_t base_src_b_stride   = src_b_stride;
                int64_t base_src_g_stride   = src_g_stride;
                int64_t base_src_icb_stride = src_icb_stride;
                int64_t his_b_stride        = dst_b_stride;
                uint64_t kernel_flags       = 0;
                if (is_first_ic) {
                    if (with_sum) {
                        base_his     = sum_src_;
                        his_b_stride = sum_src_b_stride;
                        kernel_flags |= KERNEL_FLAG_AD_BIAS();
                    } else {
                        kernel_flags |= KERNEL_FLAG_LD_BIAS();
                    }
                }
                if (is_last_ic) {
                    if (with_relu) {
                        kernel_flags |= KERNEL_FLAG_RELU();
                    } else if (with_relu6) {
                        kernel_flags |= KERNEL_FLAG_RELU6();
                    }
                }
                base_src += mbl3 * base_src_b_stride + gpl3 * base_src_g_stride + icl2 * src_h * src_w;
                base_dst += mbl3 * dst_b_stride + gpl3 * dst_g_stride;
                base_his += mbl3 * his_b_stride + gpl3 * dst_g_stride;
                base_flt += gpl3 * flt_g_stride + icl2 * sp.padded_oc;

                if (sp.down_sample) {
                    const int64_t src_trans_b_stride   = int64_t(sp.ic_l2_blk) * dst_hw;
                    const int64_t src_trans_g_stride   = int64_t(sp.mb_l3_blk) * sp.ic_l2_blk * dst_hw;
                    const int64_t src_trans_icb_stride = int64_t(dst_hw) * CH_DT_BLK();
                    float *src_trans = reinterpret_cast<float*>(temp_buffer_);
#ifdef PPL_USE_X86_OMP_COLLAPSE
                    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
#endif
                    for (int64_t g = 0; g < gpl3_eff; ++g) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
                        PRAGMA_OMP_PARALLEL_FOR()
#endif
                        for (int64_t b = 0; b < mbl3_eff; ++b) {
                            for (int64_t icb = 0; icb < div_up(icl2_eff, CH_DT_BLK()); ++icb) {
                                const float *l_base_src = base_src + g * base_src_g_stride + b * base_src_b_stride + icb * base_src_icb_stride;
                                float *l_src_trans      = src_trans + g * src_trans_g_stride + b * src_trans_b_stride + icb * src_trans_icb_stride;
                                for (int64_t ih = 0; ih < src_h; ih += cp.stride_h) {
                                    for (int64_t iw = 0; iw < src_w; iw += cp.stride_w) {
                                        memcpy32_sse(l_src_trans, l_base_src + iw * CH_DT_BLK(), CH_DT_BLK());
                                        l_src_trans += CH_DT_BLK();
                                    }
                                    l_base_src += cp.stride_h * src_h_stride;
                                }
                            }
                        }
                    }
                    base_src            = src_trans;
                    base_src_b_stride   = src_trans_b_stride;
                    base_src_g_stride   = src_trans_g_stride;
                    base_src_icb_stride = src_trans_icb_stride;
                }
                share_param[SRC_ICB_STRIDE_IDX()] = base_src_icb_stride;
                share_param[CHANNELS_IDX()] = icl2_eff;
                PICK_PARAM(uint64_t, share_param, FLAGS_IDX()) = kernel_flags;
#ifdef PPL_USE_X86_OMP_COLLAPSE
                PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(4)
#endif
                for (int64_t g = 0; g < gpl3_eff; ++g) {
                    for (int64_t b = 0; b < mbl3_eff; ++b) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
                        PRAGMA_OMP_PARALLEL_FOR()
#endif
                        for (int64_t ocl2 = 0; ocl2 < sp.padded_oc; ocl2 += sp.oc_l2_blk) {
                            for (int64_t hwl2 = 0; hwl2 < dst_hw; hwl2 += sp.hw_l2_blk) {
                                int64_t private_param[PRIV_PARAM_LEN()];
                                const int64_t ocl2_eff = min<int64_t>(sp.padded_oc - ocl2, sp.oc_l2_blk);
                                const int64_t hwl2_eff = min<int64_t>(dst_hw - hwl2, sp.hw_l2_blk);
                                const int64_t hw_body = round(hwl2_eff, sp.hw_kr_blk);
                                const int64_t hw_tail = hwl2_eff - hw_body;
                                const float *l_src  = base_src + b * base_src_b_stride + g * base_src_g_stride + hwl2 * CH_DT_BLK();
                                const float *l_his  = base_his + b * his_b_stride + g * dst_g_stride + ocl2 * dst_hw + hwl2 * CH_DT_BLK();
                                float *l_dst        = base_dst + b * dst_b_stride + g * dst_g_stride + ocl2 * dst_hw + hwl2 * CH_DT_BLK();
                                const float *l_flt  = base_flt + g * flt_g_stride + ocl2 * sp.ic_l2_blk;
                                const float *l_bias = cvt_bias_ + (g + gpl3) * sp.padded_oc + ocl2;
                                PICK_PARAM(const float *, private_param, FLT_IDX())  = l_flt;
                                PICK_PARAM(const float *, private_param, BIAS_IDX()) = l_bias;
                                for (int64_t oc = ocl2; oc < ocl2 + ocl2_eff; oc += sp.oc_kr_blk) {
                                    const int64_t oc_eff = min<int64_t>(ocl2 + ocl2_eff - oc, sp.oc_kr_blk);
                                    const int64_t oc_sel = div_up(oc_eff, CH_DT_BLK()) - 1;
                                    if (sp.hw_kr_blk == BLK1X3_HW_RF()) {
                                        if (hw_body) {
                                            PICK_PARAM(const float *, private_param, SRC_IDX())  = l_src;
                                            PICK_PARAM(const float *, private_param, HIS_IDX())  = l_his;
                                            PICK_PARAM(float *, private_param, DST_IDX())        = l_dst;
                                            private_param[HW_IDX()] = hw_body;
                                            conv2d_n8cx_gemm_direct_kernel_fp32_sse_hw3_table[nt_store_sel][oc_sel](private_param, share_param);
                                        }
                                        if (hw_tail) {
                                            PICK_PARAM(const float *, private_param, SRC_IDX())  = l_src + hw_body * CH_DT_BLK();
                                            PICK_PARAM(const float *, private_param, HIS_IDX())  = l_his + hw_body * CH_DT_BLK();
                                            PICK_PARAM(float *, private_param, DST_IDX())        = l_dst + hw_body * CH_DT_BLK();
                                            private_param[HW_IDX()] = hw_tail;
                                            conv2d_n8cx_gemm_direct_kernel_fp32_sse_hw1_table[nt_store_sel][oc_sel](private_param, share_param);
                                        }
                                    } else {
                                        PICK_PARAM(const float *, private_param, SRC_IDX())  = l_src;
                                        PICK_PARAM(const float *, private_param, HIS_IDX())  = l_his;
                                        PICK_PARAM(float *, private_param, DST_IDX())        = l_dst;
                                        private_param[HW_IDX()] = hwl2_eff;
                                        conv2d_n8cx_gemm_direct_kernel_fp32_sse_hw1_table[nt_store_sel][oc_sel](private_param, share_param);
                                    }
                                    PICK_PARAM(const float *, private_param, FLT_IDX())  += sp.oc_kr_blk * sp.ic_l2_blk;
                                    PICK_PARAM(const float *, private_param, BIAS_IDX()) += sp.oc_kr_blk;
                                    l_his += sp.oc_kr_blk * dst_hw;
                                    l_dst += sp.oc_kr_blk * dst_hw;
                                }
                            }
                        }
                    }
                }
            }
            if (sp.use_nt_store) {
                PRAGMA_OMP_PARALLEL()
                {
                    _mm_sfence();
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n8cx_gemm_direct_fp32_sse_manager::gen_cvt_weights(const float *filter, const float *bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int64_t oc_per_gp = param_.num_output / param_.group;
    const int64_t padded_oc = round_up(oc_per_gp, CH_DT_BLK());
    const int64_t ic_l2_blk = conv2d_n8cx_gemm_direct_fp32_sse_executor::cal_ic_l2_blk(param_);

    cvt_bias_size_ = param_.group * padded_oc;
    cvt_bias_      = (float *)allocator_->Alloc(cvt_bias_size_ * sizeof(float));
    if (cvt_bias_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    for (int64_t g = 0; g < param_.group; ++g) {
        memcpy(cvt_bias_ + g * padded_oc, bias + g * oc_per_gp, oc_per_gp * sizeof(float));
        memset(cvt_bias_ + g * padded_oc + oc_per_gp, 0, (padded_oc - oc_per_gp) * sizeof(float));
    }

    cvt_filter_size_ = reorder_goidhw_gIOBidhw8i8o_fp32_get_dst_size(
        param_.group, param_.num_output, param_.channels,
        1, 1, 1, ic_l2_blk);
    cvt_filter_size_ /= sizeof(float);
    cvt_filter_      = (float *)allocator_->Alloc(cvt_filter_size_ * sizeof(float));
    if (cvt_filter_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    return reorder_goidhw_gIOBidhw8i8o_fp32(
        filter, param_.group, param_.num_output, param_.channels,
        1, 1, 1, ic_l2_blk, cvt_filter_);
}

bool conv2d_n8cx_gemm_direct_fp32_sse_manager::is_supported()
{
    bool aligned_channels   = param_.channels / param_.group % CH_DT_BLK() == 0;
    bool aligned_num_output = param_.num_output / param_.group % CH_DT_BLK() == 0;
    return ((param_.group == 1) || (aligned_channels && aligned_num_output)) && param_.is_pointwise();
}

conv2d_fp32_executor *conv2d_n8cx_gemm_direct_fp32_sse_manager::gen_executor()
{
    return new conv2d_n8cx_gemm_direct_fp32_sse_executor(&param_, cvt_filter_, cvt_bias_);
}

}}}; // namespace ppl::kernel::x86
