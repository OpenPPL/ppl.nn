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

#include <string.h>

#include "ppl/kernel/x86/fp32/reorder.h"
#include "ppl/kernel/x86/common/avx_tools.h"
#include "ppl/kernel/x86/fp32/conv2d/gemm_direct/fma/conv2d_n16cx_gemm_direct_fp32_fma.h"
#include "ppl/kernel/x86/fp32/conv2d/gemm_direct/fma/conv2d_n16cx_gemm_direct_kernel_fp32_fma.h"
#include "ppl/kernel/x86/common/array_param_helper.h"
#include "ppl/common/sys.h"

namespace ppl { namespace kernel { namespace x86 {

static const int64_t ASSUME_L2_BYTES = 256 * 1024;
static const int64_t ASSUME_L2_WAYS = 4;
static const int64_t ASSUME_L3_BYTES = 2048 * 1024;
static const float L2_RATIO = 0.251f;
static const float L3_RATIO = 0.501f;

static const int64_t IC_DATA_BLK = conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::IC_DATA_BLK;
static const int64_t OC_DATA_BLK = conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_DATA_BLK;
static const int64_t OC_REG_ELTS = conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_REG_ELTS;

static const int64_t IC_L2_BLK_MAX_LARGE = 16 * IC_DATA_BLK; // preserve for tuning
static const int64_t IC_L2_BLK_MAX_SMALL = 16 * IC_DATA_BLK;
static const float IC_L2_BLK_TAIL_RATIO = 0.251f;
static const int64_t OC_L2_BLK_MAX_LARGE = 16 * OC_DATA_BLK; // preserve for tuning
static const int64_t OC_L2_BLK_MAX_SMALL = 16 * OC_DATA_BLK;
static const int64_t OC_L2_BLK_MIN = 1 * OC_DATA_BLK;
static const int64_t S_L2_BLK_MAX = 12 * conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::MAX_S_REGS;
static const int64_t S_KERNEL_BLK_MAX = conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::MAX_S_REGS;
static const int64_t S_KERNEL_BLK_MIN = conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::MAX_S_REGS - 2;

int64_t conv2d_n16cx_gemm_direct_fp32_fma_executor::cal_ic_l2_blk(const conv2d_fp32_param &param)
{
    const int64_t ic_per_grp = param.channels / param.group;
    const int64_t padded_ic = round_up(ic_per_grp, IC_DATA_BLK);
    const int64_t oc_per_grp = param.num_output / param.group;
    const int64_t padded_oc = round_up(oc_per_grp, OC_DATA_BLK);

    int64_t ic_l2_blk;
    if (padded_ic > padded_oc) {
        ic_l2_blk = min(IC_L2_BLK_MAX_LARGE, padded_ic);
    } else {
        ic_l2_blk = min(IC_L2_BLK_MAX_SMALL, padded_ic);
    }
    if (mod_up(padded_ic, ic_l2_blk) < IC_L2_BLK_TAIL_RATIO * ic_l2_blk) {
        ic_l2_blk = round_up(padded_ic / (padded_ic / ic_l2_blk), IC_DATA_BLK);
    }
    return ic_l2_blk;
}

void conv2d_n16cx_gemm_direct_fp32_fma_executor::init_preproc_param()
{
    schedule_param_.ic_per_grp = conv_param_->channels / conv_param_->group;
    schedule_param_.oc_per_grp = conv_param_->num_output / conv_param_->group;
    schedule_param_.padded_ic = round_up(schedule_param_.ic_per_grp, IC_DATA_BLK);
    schedule_param_.padded_oc = round_up(schedule_param_.oc_per_grp, OC_DATA_BLK);
}

void conv2d_n16cx_gemm_direct_fp32_fma_executor::cal_kernel_tunning_param()
{
    const conv2d_fp32_param &cp = *conv_param_;
    kernel_schedule_param &sp   = schedule_param_;

    const int64_t num_thread = PPL_OMP_MAX_THREADS();
    const int64_t batch      = src_shape_->GetDim(0);
    const int64_t dst_space  = dst_shape_->GetDim(2) * dst_shape_->GetDim(3);

    const float l3_cap_all_core = (ppl::common::GetCpuCacheL3() == 0 ? (ASSUME_L3_BYTES * num_thread) : ppl::common::GetCpuCacheL3()) * L3_RATIO / sizeof(float);

    sp.ic_l2_blk = cal_ic_l2_blk(cp);
    sp.ic_l2_cnt = div_up(sp.padded_ic, sp.ic_l2_blk);

    sp.mb_l3_blk = min(batch, num_thread);
    sp.grp_l3_blk = 1;

    sp.s_kr_blk = S_KERNEL_BLK_MAX;
#define REDUN_S(S, S_BLK) (float(round_up(S, S_BLK)) / (S)-1.0f)
    if (REDUN_S(dst_space, sp.s_kr_blk) > 0.201f) {
        for (int64_t s_blk = S_KERNEL_BLK_MAX - 1; s_blk >= S_KERNEL_BLK_MIN; --s_blk) {
            if (REDUN_S(dst_space, s_blk) < REDUN_S(dst_space, sp.s_kr_blk)) {
                sp.s_kr_blk = s_blk;
            }
        }
    }
#undef REDUN_S
    sp.s_l2_blk = min(dst_space, round_up(S_L2_BLK_MAX, sp.s_kr_blk));
    if (sp.padded_oc > sp.padded_ic) {
        sp.oc_l2_blk = min(OC_L2_BLK_MAX_LARGE, sp.padded_oc);
    } else {
        sp.oc_l2_blk = min(OC_L2_BLK_MAX_SMALL, sp.padded_oc);
    }

    const int64_t oc_thread = div_up(num_thread, sp.grp_l3_blk * sp.mb_l3_blk * div_up(dst_space, sp.s_l2_blk));
    if (sp.padded_oc / oc_thread < sp.oc_l2_blk) {
        sp.oc_l2_blk = round_up(max(sp.padded_oc / oc_thread, OC_L2_BLK_MIN), OC_DATA_BLK);
    }

    sp.down_sample = 0;
    if (cp.stride_h > 1 || cp.stride_w > 1) {
        sp.down_sample = 1;
    }

    sp.use_nt_store = 0;
    if (batch * cp.group * sp.padded_oc * dst_space > l3_cap_all_core * 3) {
        sp.use_nt_store = 1;
    }
}

uint64_t conv2d_n16cx_gemm_direct_fp32_fma_executor::cal_temp_buffer_size()
{
    if (schedule_param_.down_sample) {
        const int64_t dst_space = dst_shape_->GetDim(2) * dst_shape_->GetDim(3);
        return (uint64_t)dst_space * schedule_param_.mb_l3_blk * schedule_param_.grp_l3_blk * schedule_param_.ic_l2_blk * sizeof(float);
    }
    return 64u;
}

ppl::common::RetCode conv2d_n16cx_gemm_direct_fp32_fma_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_ || ((conv_param_->fuse_flag & conv_fuse_flag::SUM) && !sum_src_shape_)) {
        return ppl::common::RC_INVALID_VALUE;
    }

    init_preproc_param();
    cal_kernel_tunning_param();

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n16cx_gemm_direct_fp32_fma_executor::execute()
{
    if (!conv_param_ || !cvt_filter_ || !cvt_bias_ || !src_ || !dst_ || ((conv_param_->fuse_flag & conv_fuse_flag::SUM) && !sum_src_) || !temp_buffer_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const conv2d_fp32_param &cp     = *conv_param_;
    const kernel_schedule_param &sp = schedule_param_;

    const int64_t batch         = src_shape_->GetDim(0);
    const int64_t src_h         = src_shape_->GetDim(2);
    const int64_t src_w         = src_shape_->GetDim(3);
    const int64_t dst_space     = dst_shape_->GetDim(2) * dst_shape_->GetDim(3);
    const int64_t padded_reg_oc = round_up(sp.oc_per_grp, OC_REG_ELTS);

    const int64_t src_b_stride   = round_up(src_shape_->GetDim(1), IC_DATA_BLK) * src_h * src_w;
    const int64_t src_g_stride   = sp.padded_ic * src_h * src_w;
    const int64_t src_icb_stride = src_h * src_w * IC_DATA_BLK;
    const int64_t src_h_stride   = src_w * IC_DATA_BLK;
    const int64_t dst_b_stride   = round_up(dst_shape_->GetDim(1), OC_DATA_BLK) * dst_space;
    const int64_t dst_g_stride   = sp.padded_oc * dst_space;
    const int64_t flt_g_stride   = sp.ic_l2_cnt * sp.padded_oc * sp.ic_l2_blk;

    const bool with_sum   = cp.fuse_flag & conv_fuse_flag::SUM;
    const bool with_relu  = cp.fuse_flag & conv_fuse_flag::RELU;
    const bool with_relu6 = cp.fuse_flag & conv_fuse_flag::RELU6;

    int64_t sum_src_b_stride = 0;
    if (with_sum) {
        sum_src_b_stride = int64_t(round_up(sum_src_shape_->GetDim(1), OC_DATA_BLK)) * dst_space;
    }

    for (int64_t mbl3 = 0; mbl3 < batch; mbl3 += sp.mb_l3_blk) {
        const int64_t mbl3_eff = min(batch - mbl3, sp.mb_l3_blk);
        for (int64_t grpl3 = 0; grpl3 < cp.group; grpl3 += sp.grp_l3_blk) {
            const int64_t grpl3_eff = min(cp.group - grpl3, sp.grp_l3_blk);
            for (int64_t icl2 = 0; icl2 < sp.padded_ic; icl2 += sp.ic_l2_blk) {
                const int64_t icl2_eff = min(sp.ic_per_grp - icl2, sp.ic_l2_blk);
                const bool is_first_ic = icl2 == 0;
                const bool is_last_ic  = (icl2 + sp.ic_l2_blk >= sp.ic_per_grp);

                const float *base_src = src_;
                const float *base_his = dst_;
                const float *base_flt = cvt_filter_;
                float *base_dst       = dst_;

                int64_t base_src_b_stride   = src_b_stride;
                int64_t base_src_g_stride   = src_g_stride;
                int64_t base_src_icb_stride = src_icb_stride;
                int64_t his_b_stride        = dst_b_stride;
                uint64_t ker_flags          = 0;
                if (is_first_ic) {
                    if (with_sum) {
                        base_his     = sum_src_;
                        his_b_stride = sum_src_b_stride;
                        ker_flags |= conv2d_n16cx_gemm_direct_kernel_fp32_fma::flag::ADD_BIAS;
                    } else {
                        ker_flags |= conv2d_n16cx_gemm_direct_kernel_fp32_fma::flag::LOAD_BIAS;
                    }
                }
                if (is_last_ic) {
                    if (with_relu) {
                        ker_flags |= conv2d_n16cx_gemm_direct_kernel_fp32_fma::flag::RELU;
                    } else if (with_relu6) {
                        ker_flags |= conv2d_n16cx_gemm_direct_kernel_fp32_fma::flag::RELU6;
                    }
                }
                base_src += mbl3 * base_src_b_stride + grpl3 * base_src_g_stride + icl2 * src_h * src_w;
                base_dst += mbl3 * dst_b_stride + grpl3 * dst_g_stride;
                base_his += mbl3 * his_b_stride + grpl3 * dst_g_stride;
                base_flt += grpl3 * flt_g_stride + icl2 * sp.padded_oc;

                if (sp.down_sample) {
                    const int64_t src_trans_b_stride   = int64_t(sp.ic_l2_blk) * dst_space;
                    const int64_t src_trans_g_stride   = int64_t(sp.mb_l3_blk) * sp.ic_l2_blk * dst_space;
                    const int64_t src_trans_icb_stride = int64_t(dst_space) * IC_DATA_BLK;
                    float *src_trans                   = reinterpret_cast<float*>(temp_buffer_);
#ifdef PPL_USE_X86_OMP_COLLAPSE
                    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
#endif
                    for (int64_t g = 0; g < grpl3_eff; ++g) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
                        PRAGMA_OMP_PARALLEL_FOR()
#endif
                        for (int64_t b = 0; b < mbl3_eff; ++b) {
                            for (int64_t icb = 0; icb < div_up(icl2_eff, IC_DATA_BLK); ++icb) {
                                const float *l_base_src = base_src + g * base_src_g_stride + b * base_src_b_stride + icb * base_src_icb_stride;
                                float *l_src_trans      = src_trans + g * src_trans_g_stride + b * src_trans_b_stride + icb * src_trans_icb_stride;
                                for (int64_t ih = 0; ih < src_h; ih += cp.stride_h) {
                                    for (int64_t iw = 0; iw < src_w; iw += cp.stride_w) {
                                        memcpy32_avx(l_src_trans, l_base_src + iw * IC_DATA_BLK, IC_DATA_BLK);
                                        l_src_trans += IC_DATA_BLK;
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
#ifdef PPL_USE_X86_OMP_COLLAPSE
                PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(4)
#endif
                for (int64_t g = 0; g < grpl3_eff; ++g) {
                    for (int64_t b = 0; b < mbl3_eff; ++b) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
                        PRAGMA_OMP_PARALLEL_FOR()
#endif
                        for (int64_t ocl2 = 0; ocl2 < padded_reg_oc; ocl2 += sp.oc_l2_blk) {
                            for (int64_t sl2 = 0; sl2 < dst_space; sl2 += sp.s_l2_blk) {
                                int64_t ker_param[conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::LENGTH];
                                array_param_helper ker_p(ker_param);
                                conv2d_n16cx_gemm_direct_kernel_fp32_fma ker(ker_param);
                                ker_p.pick<int64_t>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::SRC_ICB_STRIDE_IDX) = base_src_icb_stride;
                                ker_p.pick<int64_t>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::CHANNELS_IDX)       = icl2_eff;
                                ker_p.pick<int64_t>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::FLAGS_IDX)          = ker_flags;

                                const int64_t ocl2_eff = min(padded_reg_oc - ocl2, sp.oc_l2_blk);
                                const int64_t sl2_eff  = min(dst_space - sl2, sp.s_l2_blk);
                                const int64_t s_body   = round(sl2_eff, sp.s_kr_blk);
                                const int64_t s_tail   = sl2_eff - s_body;

                                const float *l_src  = base_src + b * base_src_b_stride + g * base_src_g_stride + sl2 * IC_DATA_BLK;
                                const float *l_his  = base_his + b * his_b_stride + g * dst_g_stride + ocl2 * dst_space + sl2 * OC_DATA_BLK;
                                float *l_dst        = base_dst + b * dst_b_stride + g * dst_g_stride + ocl2 * dst_space + sl2 * OC_DATA_BLK;
                                const float *l_flt  = base_flt + g * flt_g_stride + ocl2 * sp.ic_l2_blk;
                                const float *l_bias = cvt_bias_ + (g + grpl3) * sp.padded_oc + ocl2;

                                ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::FLT_PTR_IDX)  = l_flt;
                                ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::BIAS_PTR_IDX) = l_bias;
                                for (int64_t oc = ocl2; oc < ocl2 + ocl2_eff; oc += OC_DATA_BLK) {
                                    const int64_t oc_eff = min(ocl2 + ocl2_eff - oc, OC_DATA_BLK);
                                    const int64_t oc_reg = div_up(oc_eff, OC_REG_ELTS);
                                    if (s_body) {
                                        ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::SRC_PTR_IDX) = l_src;
                                        ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::HIS_PTR_IDX) = l_his;
                                        ker_p.pick<float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::DST_PTR_IDX)       = l_dst;
                                        ker_p.pick<int64_t>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::SPACE_IDX)        = s_body;
                                        ker.execute(sp.use_nt_store, oc_reg, sp.s_kr_blk);
                                    }
                                    if (s_tail) {
                                        ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::SRC_PTR_IDX) = l_src + s_body * IC_DATA_BLK;
                                        ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::HIS_PTR_IDX) = l_his + s_body * OC_DATA_BLK;
                                        ker_p.pick<float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::DST_PTR_IDX)       = l_dst + s_body * OC_DATA_BLK;
                                        ker_p.pick<int64_t>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::SPACE_IDX)        = s_tail;
                                        ker.execute(sp.use_nt_store, oc_reg, s_tail);
                                    }
                                    ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::FLT_PTR_IDX)  += OC_DATA_BLK * sp.ic_l2_blk;
                                    ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::BIAS_PTR_IDX) += OC_DATA_BLK;
                                    l_his += OC_DATA_BLK * dst_space;
                                    l_dst += OC_DATA_BLK * dst_space;
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

    const int64_t oc_per_grp = param_.num_output / param_.group;
    const int64_t padded_oc = round_up(oc_per_grp, OC_DATA_BLK);
    const int64_t ic_l2_blk = conv2d_n16cx_gemm_direct_fp32_fma_executor::cal_ic_l2_blk(param_);

    cvt_bias_size_ = param_.group * padded_oc;
    cvt_bias_      = (float *)allocator_->Alloc(cvt_bias_size_ * sizeof(float));
    if (cvt_bias_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    for (int64_t g = 0; g < param_.group; ++g) {
        memcpy(cvt_bias_ + g * padded_oc, bias + g * oc_per_grp, oc_per_grp * sizeof(float));
        memset(cvt_bias_ + g * padded_oc + oc_per_grp, 0, (padded_oc - oc_per_grp) * sizeof(float));
    }

    cvt_filter_size_ = reorder_goidhw_gIOBidhw16i16o_fp32_get_dst_size(
        param_.group, param_.num_output, param_.channels,
        1, 1, 1, ic_l2_blk);
    cvt_filter_size_ /= sizeof(float);
    cvt_filter_      = (float *)allocator_->Alloc(cvt_filter_size_ * sizeof(float));
    if (cvt_filter_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    return reorder_goidhw_gIOBidhw16i16o_fp32(
        filter, param_.group, param_.num_output, param_.channels,
        1, 1, 1, ic_l2_blk, cvt_filter_);
}

bool conv2d_n16cx_gemm_direct_fp32_fma_manager::is_supported()
{
    bool aligned_channels   = param_.channels / param_.group % IC_DATA_BLK == 0;
    bool aligned_num_output = param_.num_output / param_.group % OC_DATA_BLK == 0;
    return ((param_.group == 1) || (aligned_channels && aligned_num_output)) && param_.is_pointwise();
}

conv2d_fp32_executor *conv2d_n16cx_gemm_direct_fp32_fma_manager::gen_executor()
{
    return new conv2d_n16cx_gemm_direct_fp32_fma_executor(&param_, cvt_filter_, cvt_bias_);
}

}}}; // namespace ppl::kernel::x86