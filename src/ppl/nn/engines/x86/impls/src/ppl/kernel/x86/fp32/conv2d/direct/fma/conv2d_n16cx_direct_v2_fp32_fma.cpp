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
#include "ppl/kernel/x86/fp32/conv2d/direct/fma/conv2d_n16cx_direct_v2_fp32_fma.h"
#include "ppl/kernel/x86/fp32/conv2d/direct/fma/conv2d_n16cx_direct_v2_kernel_fp32_fma.h"
#include "ppl/common/sys.h"

#define ASSUME_L2_BYTES() (256 * 1024)
#define ASSUME_L2_WAYS()  4
#define ASSUME_L3_BYTES() (2048 * 1024)
#define L2_RATIO()        0.251
#define L3_RATIO()        0.501

#define IC_L2_BLK_MAX()        (16 * CH_DT_BLK())
#define IC_L2_BLK_TAIL_RATIO() 0.334
#define OC_L2_BLK_MAX()        (4 * CH_DT_BLK())

#define PADDING_POLICY_NOPAD() 0
#define PADDING_POLICY_PREPAD() 1

namespace ppl { namespace kernel { namespace x86 {

int32_t conv2d_n16cx_direct_v2_fp32_fma_executor::cal_ic_l2_blk(const conv2d_fp32_param &param)
{
    const int32_t ic_per_gp = param.channels / param.group;
    const int32_t padded_ic = round_up(ic_per_gp, CH_DT_BLK());

    int32_t ic_l2_blk;
    if (padded_ic >= IC_L2_BLK_MAX()) {
        ic_l2_blk = min(div_up(4 * IC_L2_BLK_MAX(), param.kernel_h * param.kernel_w * CH_DT_BLK()) * CH_DT_BLK(), padded_ic);
    } else {
        ic_l2_blk = min(div_up(IC_L2_BLK_MAX(), param.kernel_h * param.kernel_w * CH_DT_BLK()) * CH_DT_BLK(), padded_ic);
    }
    if (mod_up(padded_ic, ic_l2_blk) < IC_L2_BLK_TAIL_RATIO() * ic_l2_blk) {
        ic_l2_blk = round_up(padded_ic / (padded_ic / ic_l2_blk), CH_DT_BLK());
    }

    return ic_l2_blk;
}

void conv2d_n16cx_direct_v2_fp32_fma_executor::init_preproc_param()
{
    schedule_param_.ic_per_gp = conv_param_->channels / conv_param_->group;
    schedule_param_.oc_per_gp = conv_param_->num_output / conv_param_->group;
    schedule_param_.padded_ic = round_up(schedule_param_.ic_per_gp, CH_DT_BLK());
    schedule_param_.padded_oc = round_up(schedule_param_.oc_per_gp, CH_DT_BLK());
}

void conv2d_n16cx_direct_v2_fp32_fma_executor::cal_kernel_tunning_param()
{
    const conv2d_fp32_param &cp = *conv_param_;
    kernel_schedule_param &sp   = schedule_param_;

    const int32_t num_thread   = PPL_OMP_MAX_THREADS();
    const int32_t batch        = src_shape_->GetDim(0);
    const int32_t src_h        = src_shape_->GetDim(2);
    const int32_t src_w        = src_shape_->GetDim(3);
    const int32_t dst_h        = dst_shape_->GetDim(2);
    const int32_t dst_w        = dst_shape_->GetDim(3);
    const int32_t ext_kernel_w = (cp.kernel_w - 1) * cp.dilation_w + 1;

    const float l3_cap_all_core = (ppl::common::GetCpuCacheL3() == 0 ? (ASSUME_L3_BYTES() * num_thread) : ppl::common::GetCpuCacheL3()) * L3_RATIO() / sizeof(float);

    sp.ic_l2_blk = cal_ic_l2_blk(cp);
    sp.ic_l2_cnt = div_up(sp.padded_ic, sp.ic_l2_blk);

    sp.gp_l3_blk = min<int32_t>(cp.group, num_thread);
    sp.mb_l3_blk = min<int32_t>(batch, div_up(num_thread, sp.gp_l3_blk));
    const int64_t padded_src_hw = int64_t(src_h) * (src_w + 2 * cp.pad_w);
    while (sp.gp_l3_blk > 1 && sp.gp_l3_blk * sp.mb_l3_blk * sp.ic_l2_blk * padded_src_hw > l3_cap_all_core) {
        --sp.gp_l3_blk;
    }
    sp.mb_l3_blk = min<int32_t>(batch, div_up(num_thread, sp.gp_l3_blk));
    while (sp.mb_l3_blk > 1 && sp.gp_l3_blk * sp.mb_l3_blk * sp.ic_l2_blk * padded_src_hw > l3_cap_all_core) {
        --sp.mb_l3_blk;
    }

    if (dst_h <= 112 && dst_w <= 112
        && cp.stride_w < dst_w && cp.pad_w != 0
        && cp.dilation_w < dst_w) {
        sp.padding_policy = PADDING_POLICY_PREPAD();
    } else {
        sp.padding_policy = PADDING_POLICY_NOPAD();
    }

    sp.unroll_ow_start = -1;
    sp.unroll_ow_end = -1;
    if (sp.padding_policy == PADDING_POLICY_NOPAD()) {
        for (int32_t ow = 0; ow < dst_w; ++ow) {
            if (ow * cp.stride_w - cp.pad_w >= 0) {
                sp.unroll_ow_start = ow;
                break;
            }
        }
        for (int32_t ow = dst_w - 1; ow >= 0; --ow) {
            if (ow * cp.stride_w - cp.pad_w + ext_kernel_w <= src_w) {
                sp.unroll_ow_end = ow + 1;
                break;
            }
        }
        if (sp.unroll_ow_start >= sp.unroll_ow_end || sp.unroll_ow_start < 0 || sp.unroll_ow_end < 0) {
            sp.unroll_ow_start = sp.unroll_ow_end = dst_w;
        }
    } else {
        sp.unroll_ow_start = 0;
        sp.unroll_ow_end = dst_w;
    }

    if (sp.unroll_ow_start < sp.unroll_ow_end) {
        sp.ow_kr_blk = min(sp.unroll_ow_end - sp.unroll_ow_start, MAX_OW_RF());
#define REDUN_W(W, W_BLK) (float(round_up(W, W_BLK)) / (W)-1.0f)
        if (REDUN_W(dst_w, sp.ow_kr_blk) > 0.201f) {
            for (int32_t ow_blk = MAX_OW_RF(); ow_blk >= MAX_OW_RF() - 2; --ow_blk) {
                if (REDUN_W(dst_w, ow_blk) < REDUN_W(dst_w, sp.ow_kr_blk)) {
                    sp.ow_kr_blk = ow_blk;
                }
            }
        }
#undef REDUN_W
    } else {
        sp.ow_kr_blk = MAX_OW_RF();
    }

    sp.oc_l2_blk = min(OC_L2_BLK_MAX(), sp.padded_oc);

    sp.use_nt_store = 0;
    if (batch * cp.group * sp.padded_oc * dst_h * dst_w > l3_cap_all_core * 2) {
        sp.use_nt_store = 1;
    }
}

uint64_t conv2d_n16cx_direct_v2_fp32_fma_executor::cal_temp_buffer_size()
{
    if (schedule_param_.padding_policy == PADDING_POLICY_PREPAD()) {
        const int32_t src_h          = src_shape_->GetDim(2);
        const int32_t src_w          = src_shape_->GetDim(3);
        const uint64_t padded_src_hw = uint64_t(src_h) * (src_w + 2 * conv_param_->pad_w);
        return padded_src_hw * schedule_param_.mb_l3_blk * schedule_param_.gp_l3_blk * schedule_param_.ic_l2_blk * sizeof(float);
    }
    return 64u;
}

ppl::common::RetCode conv2d_n16cx_direct_v2_fp32_fma_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_ || ((conv_param_->fuse_flag & conv_fuse_flag::sum) && !sum_src_shape_)) {
        return ppl::common::RC_INVALID_VALUE;
    }

    init_preproc_param();
    cal_kernel_tunning_param();

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n16cx_direct_v2_fp32_fma_executor::execute()
{
    if (!conv_param_ || !cvt_filter_ || !cvt_bias_ || !src_ || !dst_ || ((conv_param_->fuse_flag & conv_fuse_flag::sum) && !sum_src_) || !temp_buffer_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const conv2d_fp32_param &cp     = *conv_param_;
    const kernel_schedule_param &sp = schedule_param_;

    const int32_t batch = src_shape_->GetDim(0);
    const int32_t src_h = src_shape_->GetDim(2);
    const int32_t src_w = src_shape_->GetDim(3);
    const int32_t dst_h = dst_shape_->GetDim(2);
    const int32_t dst_w = dst_shape_->GetDim(3);

    const int32_t ext_kernel_h = (cp.kernel_h - 1) * cp.dilation_h + 1;
    const int32_t ext_kernel_w = (cp.kernel_w - 1) * cp.dilation_w + 1;
    const int64_t padded_rf_oc = round_up(sp.oc_per_gp, CH_RF_BLK());

    const int64_t src_b_stride   = int64_t(round_up(src_shape_->GetDim(1), CH_DT_BLK())) * src_h * src_w;
    const int64_t src_g_stride   = int64_t(sp.padded_ic) * src_h * src_w;
    const int64_t src_icb_stride = int64_t(src_h) * src_w * CH_DT_BLK();
    const int64_t src_h_stride   = int64_t(src_w) * CH_DT_BLK();
    const int64_t src_sw_stride  = int64_t(cp.stride_w) * CH_DT_BLK();
    const int64_t src_dh_stride  = int64_t(cp.dilation_h) * src_w * CH_DT_BLK();
    const int64_t src_dw_stride  = int64_t(cp.dilation_w) * CH_DT_BLK();
    const int64_t dst_b_stride   = int64_t(round_up(dst_shape_->GetDim(1), CH_DT_BLK())) * dst_h * dst_w;
    const int64_t dst_g_stride   = int64_t(sp.padded_oc) * dst_h * dst_w;
    const int64_t dst_h_stride   = int64_t(dst_w) * CH_DT_BLK();
    const int64_t flt_g_stride   = int64_t(sp.ic_l2_cnt) * sp.padded_oc * cp.kernel_h * cp.kernel_w * sp.ic_l2_blk;

    const bool with_sum   = cp.fuse_flag & conv_fuse_flag::sum;
    const bool with_relu  = cp.fuse_flag & conv_fuse_flag::relu;
    const bool with_relu6 = cp.fuse_flag & conv_fuse_flag::relu6;

    int64_t sum_src_b_stride = 0;
    if (with_sum) {
        sum_src_b_stride = int64_t(round_up(sum_src_shape_->GetDim(1), CH_DT_BLK())) * dst_h * dst_w;
    }

    PRAGMA_OMP_PARALLEL()
    {
    int64_t share_param[SHAR_PARAM_LEN()];
    share_param[KH_IDX()] = cp.kernel_h;
    share_param[KW_IDX()] = cp.kernel_w;
    const int64_t nt_store_sel = sp.use_nt_store;
    const int64_t stride_w_sel = cp.stride_w > 2 ? 0 : cp.stride_w;
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
                int64_t base_src_h_stride   = src_h_stride;
                int64_t base_src_dh_stride  = src_dh_stride;
                int64_t his_b_stride        = dst_b_stride;
                uint64_t kernel_flags = 0;
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
                base_flt += gpl3 * flt_g_stride + icl2 * sp.padded_oc * cp.kernel_h * cp.kernel_w;

                if (sp.padding_policy == PADDING_POLICY_PREPAD()) {
                    const int64_t src_trans_w          = src_w + 2 * cp.pad_w;
                    const int64_t src_trans_b_stride   = int64_t(sp.ic_l2_blk) * src_h * src_trans_w;
                    const int64_t src_trans_g_stride   = int64_t(sp.mb_l3_blk) * sp.ic_l2_blk * src_h * src_trans_w;
                    const int64_t src_trans_icb_stride = int64_t(src_h) * src_trans_w * CH_DT_BLK();
                    const int64_t src_trans_h_stride   = int64_t(src_trans_w) * CH_DT_BLK();
                    const int64_t src_trans_dh_stride  = int64_t(cp.dilation_h) * src_trans_w * CH_DT_BLK();
                    float *src_trans = reinterpret_cast<float*>(temp_buffer_);
#ifdef PPL_USE_X86_OMP_COLLAPSE
                    PRAGMA_OMP_FOR_COLLAPSE(3)
#endif
                    for (int64_t g = 0; g < gpl3_eff; ++g) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
                        PRAGMA_OMP_FOR()
#endif
                        for (int64_t b = 0; b < mbl3_eff; ++b) {
                            for (int64_t icb = 0; icb < div_up(icl2_eff, CH_DT_BLK()); ++icb) {
                                const float *l_base_src = base_src + g * base_src_g_stride + b * base_src_b_stride + icb * base_src_icb_stride;
                                float *l_src_trans      = src_trans + g * src_trans_g_stride + b * src_trans_b_stride + icb * src_trans_icb_stride;
                                for (int64_t ih = 0; ih < src_h; ++ih) {
                                    memset32_avx(l_src_trans, 0, cp.pad_w * CH_DT_BLK());
                                    l_src_trans += cp.pad_w * CH_DT_BLK();
                                    memcpy32_avx(l_src_trans, l_base_src, src_h_stride);
                                    l_src_trans += src_h_stride;
                                    l_base_src += src_h_stride;
                                    memset32_avx(l_src_trans, 0, cp.pad_w * CH_DT_BLK());
                                    l_src_trans += cp.pad_w * CH_DT_BLK();
                                }
                            }
                        }
                    }
                    base_src            = src_trans + cp.pad_w * CH_DT_BLK();
                    base_src_b_stride   = src_trans_b_stride;
                    base_src_g_stride   = src_trans_g_stride;
                    base_src_icb_stride = src_trans_icb_stride;
                    base_src_h_stride   = src_trans_h_stride;
                    base_src_dh_stride  = src_trans_dh_stride;
                }
                share_param[SRC_ICB_STRIDE_IDX()] = base_src_icb_stride;
                share_param[SRC_SW_STRIDE_IDX()] = src_sw_stride;
                share_param[SRC_DH_STRIDE_IDX()] = base_src_dh_stride;
                share_param[SRC_DW_STRIDE_IDX()] = src_dw_stride;
                share_param[CHANNELS_IDX()] = icl2_eff;
                PICK_PARAM(uint64_t, share_param, FLAGS_IDX()) = kernel_flags;
#ifdef PPL_USE_X86_OMP_COLLAPSE
                PRAGMA_OMP_FOR_COLLAPSE(4)
#endif
                for (int64_t g = 0; g < gpl3_eff; ++g) {
                    for (int64_t b = 0; b < mbl3_eff; ++b) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
                        PRAGMA_OMP_FOR()
#endif
                        for (int64_t ocl2 = 0; ocl2 < padded_rf_oc; ocl2 += sp.oc_l2_blk) {
                            for (int64_t oh = 0; oh < dst_h; ++oh) {
                                int64_t private_param[PRIV_PARAM_LEN()];
                                const int64_t ocl2_eff = min<int64_t>(padded_rf_oc - ocl2, sp.oc_l2_blk);
                                const int64_t ih       = oh * cp.stride_h - cp.pad_h;
                                private_param[KH_START_IDX()] = div_up(min<int64_t>(max<int64_t>(0 - ih, 0), ext_kernel_h - 1), cp.dilation_h);
                                private_param[KH_END_IDX()]   = div_up(max<int64_t>(min<int64_t>(src_h - ih, ext_kernel_h), 0), cp.dilation_h);
                                const int64_t ow_unroll_len  = sp.unroll_ow_end - sp.unroll_ow_start;
                                const int64_t ow_unroll_body = round(ow_unroll_len, sp.ow_kr_blk);
                                const int64_t ow_unroll_tail = ow_unroll_len - ow_unroll_body;
                                const float *l_src  = base_src + b * base_src_b_stride + g * base_src_g_stride + ih * base_src_h_stride - cp.pad_w * CH_DT_BLK();
                                const float *l_his  = base_his + b * his_b_stride + g * dst_g_stride + ocl2 * dst_h * dst_w + oh * dst_h_stride;
                                float *l_dst        = base_dst + b * dst_b_stride + g * dst_g_stride + ocl2 * dst_h * dst_w + oh * dst_h_stride;
                                const float *l_flt  = base_flt + g * flt_g_stride + ocl2 * sp.ic_l2_blk * cp.kernel_h * cp.kernel_w;
                                const float *l_bias = cvt_bias_ + (g + gpl3) * sp.padded_oc + ocl2;
                                for (int64_t oc = ocl2; oc < ocl2 + ocl2_eff; oc += CH_DT_BLK()) {
                                    const int64_t oc_eff = min<int64_t>(ocl2 + ocl2_eff - oc, CH_DT_BLK());
                                    const int64_t oc_sel = div_up(oc_eff, CH_RF_BLK()) - 1;

                                    PICK_PARAM(const float *, private_param, SRC_IDX())  = l_src;
                                    PICK_PARAM(const float *, private_param, HIS_IDX())  = l_his;
                                    PICK_PARAM(float *, private_param, DST_IDX())        = l_dst;
                                    PICK_PARAM(const float *, private_param, FLT_IDX())  = l_flt;
                                    PICK_PARAM(const float *, private_param, BIAS_IDX()) = l_bias;

                                    for (int64_t ow = 0; ow < sp.unroll_ow_start; ++ow) {
                                        const int64_t iw              = ow * cp.stride_w - cp.pad_w;
                                        if (cp.dilation_w == 1) {
                                            private_param[KW_START_IDX()] = min<int64_t>(max<int64_t>(0 - iw, 0), ext_kernel_w - 1);
                                            private_param[KW_END_IDX()]   = max<int64_t>(min<int64_t>(src_w - iw, ext_kernel_w), 0);
                                        } else {
                                            private_param[KW_START_IDX()] = div_up(min<int64_t>(max<int64_t>(0 - iw, 0), ext_kernel_w - 1), cp.dilation_w);
                                            private_param[KW_END_IDX()]   = div_up(max<int64_t>(min<int64_t>(src_w - iw, ext_kernel_w), 0), cp.dilation_w);
                                        }
                                        conv2d_n16cx_direct_v2_kernel_fp32_fma_pad_table[nt_store_sel][oc_sel](private_param, share_param);
                                        PICK_PARAM(const float *, private_param, SRC_IDX()) += src_sw_stride;
                                        PICK_PARAM(const float *, private_param, HIS_IDX()) += CH_DT_BLK();
                                        PICK_PARAM(float *, private_param, DST_IDX()) += CH_DT_BLK();
                                    }

                                    if (ow_unroll_body) {
                                        private_param[OW_IDX()] = ow_unroll_body;
                                        conv2d_n16cx_direct_v2_kernel_fp32_fma_blk_table[nt_store_sel][stride_w_sel][oc_sel][sp.ow_kr_blk - 1](private_param, share_param);
                                        PICK_PARAM(const float *, private_param, SRC_IDX()) += ow_unroll_body * src_sw_stride;
                                        PICK_PARAM(const float *, private_param, HIS_IDX()) += ow_unroll_body * CH_DT_BLK();
                                        PICK_PARAM(float *, private_param, DST_IDX()) += ow_unroll_body * CH_DT_BLK();
                                    }
                                    if (ow_unroll_tail) {
                                        private_param[OW_IDX()] = ow_unroll_tail;
                                        conv2d_n16cx_direct_v2_kernel_fp32_fma_blk_table[nt_store_sel][stride_w_sel][oc_sel][ow_unroll_tail - 1](private_param, share_param);
                                        PICK_PARAM(const float *, private_param, SRC_IDX()) += ow_unroll_tail * src_sw_stride;
                                        PICK_PARAM(const float *, private_param, HIS_IDX()) += ow_unroll_tail * CH_DT_BLK();
                                        PICK_PARAM(float *, private_param, DST_IDX()) += ow_unroll_tail * CH_DT_BLK();
                                    }

                                    for (int64_t ow = sp.unroll_ow_end; ow < dst_w; ++ow) {
                                        const int64_t iw              = ow * cp.stride_w - cp.pad_w;
                                        if (cp.dilation_w == 1) {
                                            private_param[KW_START_IDX()] = min<int64_t>(max<int64_t>(0 - iw, 0), ext_kernel_w - 1);
                                            private_param[KW_END_IDX()]   = max<int64_t>(min<int64_t>(src_w - iw, ext_kernel_w), 0);
                                        } else {
                                            private_param[KW_START_IDX()] = div_up(min<int64_t>(max<int64_t>(0 - iw, 0), ext_kernel_w - 1), cp.dilation_w);
                                            private_param[KW_END_IDX()]   = div_up(max<int64_t>(min<int64_t>(src_w - iw, ext_kernel_w), 0), cp.dilation_w);
                                        }
                                        conv2d_n16cx_direct_v2_kernel_fp32_fma_pad_table[nt_store_sel][oc_sel](private_param, share_param);
                                        PICK_PARAM(const float *, private_param, SRC_IDX()) += src_sw_stride;
                                        PICK_PARAM(const float *, private_param, HIS_IDX()) += CH_DT_BLK();
                                        PICK_PARAM(float *, private_param, DST_IDX()) += CH_DT_BLK();
                                    }
                                    l_bias += CH_DT_BLK();
                                    l_flt  += CH_DT_BLK() * sp.ic_l2_blk * cp.kernel_h * cp.kernel_w;
                                    l_dst  += CH_DT_BLK() * dst_h * dst_w;
                                    l_his  += CH_DT_BLK() * dst_h * dst_w;
                                }
                            }
                        }
                    }
                }
            }
            if (sp.use_nt_store) {
                _mm_sfence();
            }
        }
    }
    } // OMP_PARALLEL

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n16cx_direct_v2_fp32_fma_manager::gen_cvt_weights(const float *filter, const float *bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int32_t oc_per_gp = param_.num_output / param_.group;
    const int32_t padded_oc = round_up(oc_per_gp, CH_DT_BLK());
    const int32_t ic_l2_blk = conv2d_n16cx_direct_v2_fp32_fma_executor::cal_ic_l2_blk(param_);

    cvt_bias_size_ = param_.group * padded_oc;
    cvt_bias_      = (float *)allocator_->Alloc(cvt_bias_size_ * sizeof(float));
    if (cvt_bias_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    for (int32_t g = 0; g < param_.group; ++g) {
        memcpy(cvt_bias_ + g * padded_oc, bias + g * oc_per_gp, oc_per_gp * sizeof(float));
        memset(cvt_bias_ + g * padded_oc + oc_per_gp, 0, (padded_oc - oc_per_gp) * sizeof(float));
    }

    cvt_filter_size_ = reorder_goidhw_gIOBidhw16i16o_fp32_get_dst_size(
        param_.group, param_.num_output, param_.channels,
        1, param_.kernel_h, param_.kernel_w, ic_l2_blk);
    cvt_filter_size_ /= sizeof(float);
    cvt_filter_      = (float *)allocator_->Alloc(cvt_filter_size_ * sizeof(float));
    if (cvt_filter_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    return reorder_goidhw_gIOBidhw16i16o_fp32(
        filter, param_.group, param_.num_output, param_.channels,
        1, param_.kernel_h, param_.kernel_w, ic_l2_blk, cvt_filter_);
}

bool conv2d_n16cx_direct_v2_fp32_fma_manager::is_supported()
{
    if (param_.is_pointwise()) {
        return false;
    }
    bool aligned_channels   = param_.channels / param_.group % CH_DT_BLK() == 0;
    bool aligned_num_output = param_.num_output / param_.group % CH_DT_BLK() == 0;
    return (param_.group == 1) || (aligned_channels && aligned_num_output);
}

conv2d_fp32_executor *conv2d_n16cx_direct_v2_fp32_fma_manager::gen_executor()
{
    return new conv2d_n16cx_direct_v2_fp32_fma_executor(&param_, cvt_filter_, cvt_bias_);
}

}}}; // namespace ppl::kernel::x86
