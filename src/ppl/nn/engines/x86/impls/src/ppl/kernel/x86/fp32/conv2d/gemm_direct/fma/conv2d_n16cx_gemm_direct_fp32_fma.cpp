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
#include <immintrin.h>
#include <string.h>

#include "ppl/kernel/x86/fp32/reorder.h"
#include "ppl/kernel/x86/common/avx_tools.h"
#include "ppl/kernel/x86/fp32/conv2d/gemm_direct/fma/conv2d_n16cx_gemm_direct_fp32_fma.h"
#include "ppl/kernel/x86/fp32/conv2d/gemm_direct/fma/conv2d_n16cx_gemm_direct_kernel_fp32_fma.h"
#include "ppl/common/sys.h"

#define ASSUME_L2_BYTES() (256 * 1024)
#define ASSUME_L2_WAYS()  4
#define ASSUME_L3_BYTES() (2048 * 1024)
#define L2_RATIO()        0.251
#define L3_RATIO()        0.501

#define HW_KR_BLK_MIN() (BLK1X6_HW_RF() - 2)
#define HW_KR_BLK_MAX() BLK1X6_HW_RF()

#define HW_L3_BLK_MAX(SP)      (1024 * (SP).hw_kr_blk)
#define HW_L3_BLK_TAIL_RATIO() 0.251
#define HW_L2_BLK_MAX(SP)      (32 * (SP).hw_kr_blk)
#define HW_L2_BLK_MIN(SP)      (4 * (SP).hw_kr_blk)
#define HW_L2_BLK_TAIL_RATIO() 0.667
#define IC_L2_BLK_MAX()        (16 * CH_DT_BLK())
#define IC_L2_BLK_TAIL_RATIO() 0.334
#define OC_L2_BLK_MAX()        (8 * CH_DT_BLK())
#define OC_L2_BLK_MIN()        (1 * CH_DT_BLK())
#define OC_L2_GRP_MIN()        4
#define OC_L2_BLK_TAIL_RATIO() 0 // 0.667
#define OC_UTILITY_MIN()       (16 * CH_DT_BLK())
#define THREAD_TAIL_RATIO()    0.8
#define THREAD_BODY_ROUND()    4

namespace ppl { namespace kernel { namespace x86 {

int32_t conv2d_n16cx_gemm_direct_fp32_fma_executor::cal_ic_l2_blk(const conv2d_fp32_param &param)
{
    const int32_t ic_per_gp = param.channels / param.group;
    const int32_t padded_ic = round_up(ic_per_gp, CH_DT_BLK());

    int32_t ic_l2_blk = min(IC_L2_BLK_MAX(), padded_ic);
    if (mod_up(padded_ic, ic_l2_blk) < IC_L2_BLK_TAIL_RATIO() * ic_l2_blk) {
        ic_l2_blk = round_up(padded_ic / (padded_ic / ic_l2_blk), CH_DT_BLK());
    }

    return ic_l2_blk;
}

void conv2d_n16cx_gemm_direct_fp32_fma_executor::init_preproc_param()
{
    schedule_param_.ic_per_gp = conv_param_->channels / conv_param_->group;
    schedule_param_.oc_per_gp = conv_param_->num_output / conv_param_->group;
    schedule_param_.padded_ic = round_up(schedule_param_.ic_per_gp, CH_DT_BLK());
    schedule_param_.padded_oc = round_up(schedule_param_.oc_per_gp, CH_DT_BLK());

    const int64_t dst_hw      = dst_shape_->GetDim(2) * dst_shape_->GetDim(3);
    schedule_param_.hw_kr_blk = HW_KR_BLK_MAX();
#define REDUN_HW(HW, HW_BLK) (float(round_up(HW, HW_BLK)) / (HW)-1.0f)
    if (REDUN_HW(dst_hw, schedule_param_.hw_kr_blk) > 0.201f) {
        for (int32_t hw_blk = HW_KR_BLK_MAX() - 1; hw_blk >= HW_KR_BLK_MIN(); --hw_blk) {
            if (REDUN_HW(dst_hw, hw_blk) < REDUN_HW(dst_hw, schedule_param_.hw_kr_blk)) {
                schedule_param_.hw_kr_blk = hw_blk;
            }
        }
    }
#undef REDUN_HW
    schedule_param_.oc_kr_blk = BLK1X6_OC_RF() * CH_RF_BLK();
    schedule_param_.cur_batch = 0;
    schedule_param_.cur_group = 0;
}

void conv2d_n16cx_gemm_direct_fp32_fma_executor::cal_kernel_tunning_param()
{
    const conv2d_fp32_param &cp = *conv_param_;
    kernel_schedule_param &sp   = schedule_param_;

    const int32_t num_thread = PPL_OMP_MAX_THREADS();
    const int32_t batch      = src_shape_->GetDim(0);
    const int32_t dst_hw     = dst_shape_->GetDim(2) * dst_shape_->GetDim(3);

    const float l2_cap_per_core = (ppl::common::GetCpuCacheL2() == 0 ? ASSUME_L2_BYTES() : ppl::common::GetCpuCacheL2()) * L2_RATIO() / sizeof(float);
    const float l3_cap_all_core = (ppl::common::GetCpuCacheL3() == 0 ? (ASSUME_L3_BYTES() * num_thread) : ppl::common::GetCpuCacheL3()) * L3_RATIO() / sizeof(float);

    sp.ic_l2_blk = cal_ic_l2_blk(cp);
    sp.ic_l2_cnt = div_up(sp.padded_ic, sp.ic_l2_blk);

    const int32_t oc_l2_blk_by_space = round_up(int32_t(l2_cap_per_core / sp.ic_l2_blk), CH_DT_BLK());
    sp.oc_l2_blk                     = min(min(max(oc_l2_blk_by_space, OC_L2_BLK_MIN()), OC_L2_BLK_MAX()), sp.padded_oc);
    if (mod_up(sp.padded_oc, sp.oc_l2_blk) < OC_L2_BLK_TAIL_RATIO() * sp.oc_l2_blk) {
        sp.oc_l2_blk = round_up(sp.padded_oc / max(sp.padded_oc / sp.oc_l2_blk, 1), CH_DT_BLK());
    }

    if (sp.padded_ic > sp.padded_oc && sp.padded_oc <= OC_L2_BLK_MAX()) {
        const int32_t num_oc_group = min(num_thread, OC_L2_GRP_MIN());
        while (num_oc_group > div_up(sp.padded_oc, sp.oc_l2_blk) && sp.oc_l2_blk > OC_L2_BLK_MIN()) {
            sp.oc_l2_blk -= CH_DT_BLK();
        }
    }

    sp.mb_l3_blk = min(batch, num_thread);
    sp.gp_l3_blk = 1;

    sp.hw_l3_blk = round_up(dst_hw, sp.hw_kr_blk);
    if (num_thread <= sp.mb_l3_blk * 2) {
        const int32_t mini_filter_factor = static_cast<int32_t>(sp.padded_ic <= CH_DT_BLK() && sp.padded_oc <= CH_DT_BLK()) + 1;
        const int32_t scaled_hw_l2_blk_max = HW_L3_BLK_MAX(sp) * mini_filter_factor;
        if (sp.hw_l3_blk > scaled_hw_l2_blk_max) {
            sp.hw_l3_blk = scaled_hw_l2_blk_max;
        }
    } else {
        const int32_t mini_filter_factor = (sp.padded_oc <= max(div_up(num_thread, sp.mb_l3_blk), 8) * CH_DT_BLK()) ? (div_up(num_thread, sp.mb_l3_blk) * 2) : 1;
        const int32_t scaled_hw_l2_blk_max = HW_L3_BLK_MAX(sp) * mini_filter_factor;
        if (sp.hw_l3_blk > scaled_hw_l2_blk_max) {
            sp.hw_l3_blk = scaled_hw_l2_blk_max;
        }
    }
    if (mod_up(dst_hw, sp.hw_l3_blk) < HW_L3_BLK_TAIL_RATIO() * sp.hw_l3_blk) {
        sp.hw_l3_blk = round_up(dst_hw / max(dst_hw / sp.hw_l3_blk, 1), sp.hw_kr_blk);
    }

    sp.use_nt_store = 0;
    if (batch * cp.group * sp.padded_oc * dst_hw > l3_cap_all_core * 2) {
        sp.use_nt_store = 1;
    }
}

void conv2d_n16cx_gemm_direct_fp32_fma_executor::cal_kernel_threading_param(const int32_t batch, const int32_t group)
{
    kernel_schedule_param &sp = schedule_param_;

    const bool bg_changed = sp.cur_batch != batch || sp.cur_group != group;
    sp.cur_batch          = batch;
    sp.cur_group          = group;

    if (bg_changed) {
        sp.hw_l2_blk = min(round_up(sp.hw_l3_blk, sp.hw_kr_blk), HW_L2_BLK_MAX(sp));
        // Split the block for load-balancing
#ifdef PPL_USE_X86_OMP_COLLAPSE
        const int32_t num_thread = PPL_OMP_MAX_THREADS();
        const int32_t num_tasks = batch * group * div_up(sp.padded_oc, sp.oc_l2_blk);
        while (
            (num_thread > num_tasks * div_up(sp.hw_l3_blk, sp.hw_l2_blk) ||
             mod_up(num_tasks * div_up(sp.hw_l3_blk, sp.hw_l2_blk), num_thread) < THREAD_TAIL_RATIO() * num_thread) &&
            (sp.hw_l2_blk > sp.hw_kr_blk)) {
            sp.hw_l2_blk -= sp.hw_kr_blk;
        }
#endif
        if (sp.padded_ic > sp.padded_oc && sp.padded_oc <= OC_UTILITY_MIN()) {
            sp.hw_l2_blk = min(sp.hw_l2_blk, HW_L2_BLK_MIN(sp));
        }
        sp.hw_l2_blk = min(sp.hw_l3_blk, sp.hw_l2_blk);
    }
}

uint64_t conv2d_n16cx_gemm_direct_fp32_fma_executor::cal_temp_buffer_size()
{
    return 64u;
}

ppl::common::RetCode conv2d_n16cx_gemm_direct_fp32_fma_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_ || ((conv_param_->fuse_flag & conv_fuse_flag::sum) && !sum_src_shape_)) {
        return ppl::common::RC_INVALID_VALUE;
    }

    init_preproc_param();
    cal_kernel_tunning_param();
    cal_kernel_threading_param(schedule_param_.mb_l3_blk, schedule_param_.gp_l3_blk);

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n16cx_gemm_direct_fp32_fma_executor::execute()
{
    if (!conv_param_ || !cvt_filter_ || !cvt_bias_ || !src_ || !dst_ || ((conv_param_->fuse_flag & conv_fuse_flag::sum) && !sum_src_) || !temp_buffer_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const conv2d_fp32_param &cp     = *conv_param_;
    const kernel_schedule_param &sp = schedule_param_;

    const int32_t batch  = src_shape_->GetDim(0);
    const int32_t src_hw = src_shape_->GetDim(2) * src_shape_->GetDim(3);
    const int32_t dst_hw = dst_shape_->GetDim(2) * dst_shape_->GetDim(3);

    const int32_t padded_src_c = round_up(src_shape_->GetDim(1), CH_DT_BLK());
    const int32_t padded_dst_c = round_up(dst_shape_->GetDim(1), CH_DT_BLK());
    const int64_t padded_rf_oc = round_up(sp.oc_per_gp, CH_RF_BLK());

    const int64_t src_g_stride   = int64_t(sp.padded_ic) * src_hw;
    const int64_t src_b_stride   = int64_t(padded_src_c) * src_hw;
    const int64_t src_icb_stride = int64_t(src_hw) * CH_DT_BLK();
    const int64_t dst_g_stride   = int64_t(sp.padded_oc) * dst_hw;
    const int64_t dst_b_stride   = int64_t(padded_dst_c) * dst_hw;
    const int64_t flt_g_stride   = int64_t(sp.ic_l2_cnt) * sp.padded_oc * sp.ic_l2_blk;
    const int64_t bias_g_stride  = sp.padded_oc;

    const bool with_sum   = cp.fuse_flag & conv_fuse_flag::sum;
    const bool with_relu  = cp.fuse_flag & conv_fuse_flag::relu;
    const bool with_relu6 = cp.fuse_flag & conv_fuse_flag::relu6;

    int64_t sum_src_b_stride = 0;
    if (with_sum) {
        sum_src_b_stride = int64_t(round_up(sum_src_shape_->GetDim(1), CH_DT_BLK())) * dst_hw;
    }

    int64_t share_param[SHAR_PARAM_LEN()];
    PICK_PARAM(float, share_param, SIX_IDX()) = 6.0f;
    for (int64_t gpl3 = 0; gpl3 < cp.group; gpl3 += sp.gp_l3_blk) {
        const int64_t gpl3_eff = min<int64_t>(cp.group - gpl3, sp.gp_l3_blk);
        for (int64_t mbl3 = 0; mbl3 < batch; mbl3 += sp.mb_l3_blk) {
            const int64_t mbl3_eff = min<int64_t>(batch - mbl3, sp.mb_l3_blk);
            for (int64_t icl2 = 0; icl2 < sp.ic_per_gp; icl2 += sp.ic_l2_blk) {
                const int64_t icl2_eff = min<int64_t>(sp.ic_per_gp - icl2, sp.ic_l2_blk);
                const bool is_first_ic = icl2 == 0;
                const bool is_last_ic  = (icl2 + sp.ic_l2_blk >= sp.ic_per_gp);

                cal_kernel_threading_param(mbl3_eff, gpl3_eff);

                const float *base_src       = src_;
                const float *base_his       = dst_;
                const float *base_flt       = cvt_filter_;
                float *base_dst             = dst_;
                int64_t base_src_b_stride   = src_b_stride;
                int64_t base_src_g_stride   = src_g_stride;
                int64_t base_src_icb_stride = src_icb_stride;
                int64_t base_his_b_stride   = dst_b_stride;
                int64_t base_dst_b_stride   = dst_b_stride;
                uint64_t kernel_flags       = 0;
                if (is_first_ic) {
                    if (with_sum) {
                        base_his     = sum_src_;
                        base_dst_b_stride = sum_src_b_stride;
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

                base_src += mbl3 * base_src_b_stride + gpl3 * base_src_g_stride + icl2 * src_hw;
                base_his += mbl3 * base_his_b_stride + gpl3 * dst_g_stride;
                base_dst += mbl3 * base_dst_b_stride + gpl3 * dst_g_stride;
                base_flt += gpl3 * flt_g_stride + icl2 * sp.padded_oc;
                for (int64_t hwl3 = 0; hwl3 < dst_hw; hwl3 += sp.hw_l3_blk) {
                    const int64_t hwl3_eff      = min<int64_t>(dst_hw - hwl3, sp.hw_l3_blk);
                    const float *tile_src       = base_src + hwl3 * CH_DT_BLK();
                    const float *tile_his       = base_his + hwl3 * CH_DT_BLK();
                    float *tile_dst             = base_dst + hwl3 * CH_DT_BLK();
                    const int64_t tile_src_b_stride   = base_src_b_stride;
                    const int64_t tile_src_g_stride   = base_src_g_stride;
                    const int64_t tile_src_icb_stride = base_src_icb_stride;
                    share_param[CHANNELS_IDX()]       = icl2_eff;
                    share_param[FLAGS_IDX()]          = kernel_flags;
                    share_param[SRC_ICB_STRIDE_IDX()] = tile_src_icb_stride;
#ifdef PPL_USE_X86_OMP_COLLAPSE
                    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(4)
#endif
                    for (int64_t ocl2 = 0; ocl2 < padded_rf_oc; ocl2 += sp.oc_l2_blk) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
                        PRAGMA_OMP_PARALLEL_FOR()
#endif
                        for (int64_t b = 0; b < mbl3_eff; ++b) {
                            for (int64_t g = 0; g < gpl3_eff; ++g) {
                                for (int64_t hwl2 = 0; hwl2 < hwl3_eff; hwl2 += sp.hw_l2_blk) {
                                    int64_t kernel_param[PRIV_PARAM_LEN()];
                                    const int64_t ocl2_eff     = min<int64_t>(padded_rf_oc - ocl2, sp.oc_l2_blk);
                                    const int64_t hwl2_eff     = min<int64_t>(hwl3_eff - hwl2, sp.hw_l2_blk);
                                    const int64_t hw_body      = round(hwl2_eff, sp.hw_kr_blk);
                                    const int64_t hw_tail      = hwl2_eff - hw_body;
                                    const int64_t nt_store_sel = sp.use_nt_store;
                                    PICK_PARAM(const float*, kernel_param, FLT_IDX()) = base_flt + g * flt_g_stride + ocl2 * sp.ic_l2_blk;
                                    PICK_PARAM(const float*, kernel_param, BIAS_IDX()) = cvt_bias_ + (gpl3 + g) * bias_g_stride + ocl2;
                                    for (int64_t oc = ocl2; oc < ocl2 + ocl2_eff; oc += CH_DT_BLK()) {
                                        const int64_t oc_eff = min<int64_t>(ocl2 + ocl2_eff - oc, CH_DT_BLK());
                                        const int64_t oc_sel = div_up(oc_eff, CH_RF_BLK()) - 1;
                                        const float *l_src   = tile_src + b * tile_src_b_stride + g * tile_src_g_stride + hwl2 * CH_DT_BLK();
                                        const float *l_his   = tile_his + b * base_his_b_stride + g * dst_g_stride + oc * dst_hw + hwl2 * CH_DT_BLK();
                                        float *l_dst         = tile_dst + b * base_dst_b_stride + g * dst_g_stride + oc * dst_hw + hwl2 * CH_DT_BLK();
                                        if (hw_body) {
                                            PICK_PARAM(const float*, kernel_param, SRC_IDX()) = l_src;
                                            PICK_PARAM(const float*, kernel_param, HIS_IDX()) = l_his;
                                            PICK_PARAM(float*, kernel_param, DST_IDX())       = l_dst;
                                            kernel_param[HW_IDX()] = hw_body;
                                            conv2d_n16cx_gemm_direct_kernel_fp32_fma_table[nt_store_sel][oc_sel][sp.hw_kr_blk - 1](kernel_param, share_param);
                                        }
                                        if (hw_tail) {
                                            PICK_PARAM(const float*, kernel_param, SRC_IDX()) = l_src + hw_body * CH_DT_BLK();
                                            PICK_PARAM(const float*, kernel_param, HIS_IDX()) = l_his + hw_body * CH_DT_BLK();
                                            PICK_PARAM(float*, kernel_param, DST_IDX())       = l_dst + hw_body * CH_DT_BLK();
                                            kernel_param[HW_IDX()] = hw_tail;
                                            conv2d_n16cx_gemm_direct_kernel_fp32_fma_table[nt_store_sel][oc_sel][hw_tail - 1](kernel_param, share_param);
                                        }
                                        PICK_PARAM(const float*, kernel_param, FLT_IDX()) += CH_DT_BLK() * sp.ic_l2_blk;
                                        PICK_PARAM(const float*, kernel_param, BIAS_IDX()) += CH_DT_BLK();
                                    }
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

ppl::common::RetCode conv2d_n16cx_gemm_direct_fp32_fma_manager::gen_cvt_weights(const float *filter, const float *bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int32_t oc_per_gp = param_.num_output / param_.group;
    const int32_t padded_oc = round_up(oc_per_gp, CH_DT_BLK());
    const int32_t ic_l2_blk = conv2d_n16cx_gemm_direct_fp32_fma_executor::cal_ic_l2_blk(param_);

    cvt_bias_size_ = param_.group * padded_oc;
    cvt_bias_      = (float *)allocator_->Alloc(cvt_bias_size_ * sizeof(float));
    if (cvt_bias_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    for (int32_t g = 0; g < param_.group; ++g) {
        memcpy(cvt_bias_ + g * padded_oc, bias + g * oc_per_gp, oc_per_gp * sizeof(float));
        memset(cvt_bias_ + g * padded_oc + oc_per_gp, 0, (padded_oc - oc_per_gp) * sizeof(float));
    }

    cvt_filter_size_ = reorder_goidhw_gIOdhwB16i16o_fp32_get_dst_size(
        param_.group, param_.num_output, param_.channels, 1, 1, 1, ic_l2_blk);
    cvt_filter_size_ /= sizeof(float);
    cvt_filter_      = (float *)allocator_->Alloc(cvt_filter_size_ * sizeof(float));
    if (cvt_filter_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    return reorder_goidhw_gIOdhwB16i16o_fp32(
        filter, param_.group, param_.num_output, param_.channels, 1, 1, 1, ic_l2_blk, cvt_filter_);
}

bool conv2d_n16cx_gemm_direct_fp32_fma_manager::is_supported()
{
    bool aligned_channels   = param_.channels / param_.group % CH_DT_BLK() == 0;
    bool aligned_num_output = param_.num_output / param_.group % CH_DT_BLK() == 0;
    return ((param_.group == 1) || (aligned_channels && aligned_num_output)) && param_.is_pointwise() && param_.stride_h == 1 && param_.stride_w == 1;
}

conv2d_fp32_executor *conv2d_n16cx_gemm_direct_fp32_fma_manager::gen_executor()
{
    return new conv2d_n16cx_gemm_direct_fp32_fma_executor(&param_, cvt_filter_, cvt_bias_);
}

}}}; // namespace ppl::kernel::x86
