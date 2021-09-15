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
#include <math.h>
#include <string.h>

#include "ppl/kernel/x86/fp32/reorder.h"
#include "ppl/kernel/x86/common/sse_tools.h"
#include "ppl/kernel/x86/fp32/conv2d/im2col_gemm/sse/conv2d_im2col_gemm_fp32_sse.h"
#include "ppl/kernel/x86/fp32/conv2d/im2col_gemm/sse/conv_gemm_kernel_fp32_sse.h"
#include "ppl/kernel/x86/common/array_param_helper.h"
#include "ppl/common/sys.h"

namespace ppl { namespace kernel { namespace x86 {

static const int64_t assume_l2_bytes = 256 * 1024;
static const int64_t assume_l2_ways = 4;
static const int64_t assume_l3_bytes = 2048 * 1024;
static const float l2_ratio = 0.251f;
static const float l3_ratio = 0.501f;

static const int64_t k_l2_blk_max = 12 * conv_gemm_kernel_fp32_sse::config::unroll_k;
static const int64_t hw_ker_blk_max = conv_gemm_kernel_fp32_sse::config::max_n_blk;
static const int64_t hw_regb_elts = conv_gemm_kernel_fp32_sse::config::n_regb_elts;
static const int64_t hw_reg_elts = conv_gemm_kernel_fp32_sse::config::n_reg_elts;
static const int64_t hw_l2_blk_max = 2 * hw_ker_blk_max;


void conv2d_im2col_gemm_fp32_sse_executor::init_preproc_param()
{
    schedule_param_.ic_per_gp = conv_param_->channels / conv_param_->group;
    schedule_param_.oc_per_gp = conv_param_->num_output / conv_param_->group;
    schedule_param_.k_per_gp = schedule_param_.ic_per_gp * conv_param_->kernel_h * conv_param_->kernel_w;
    schedule_param_.padded_k = round_up(schedule_param_.k_per_gp, conv_gemm_kernel_fp32_sse::config::unroll_k);
}

void conv2d_im2col_gemm_fp32_sse_executor::cal_kernel_tunning_param()
{
    const conv2d_fp32_param &cp = *conv_param_;
    kernel_schedule_param &sp   = schedule_param_;

    const int32_t num_thread = PPL_OMP_MAX_THREADS();
    const int32_t batch      = src_shape_->GetDim(0);
    const int32_t dst_hw     = dst_shape_->GetDim(2) * dst_shape_->GetDim(3);

    const float l3_cap_all_core = (ppl::common::GetCpuCacheL3() == 0 ? (assume_l3_bytes * num_thread) : ppl::common::GetCpuCacheL3()) * l3_ratio / sizeof(float);

    sp.hw_l2_blk = round_up(min<int64_t>(dst_hw, hw_l2_blk_max), hw_ker_blk_max);

    sp.mb_l3_blk = 1;
    sp.gp_l3_blk = 1;
    const int64_t im2col_size_per_img = sp.k_per_gp * dst_hw;
    while ((sp.mb_l3_blk + 1) * sp.gp_l3_blk * im2col_size_per_img < l3_cap_all_core && sp.mb_l3_blk < batch && sp.mb_l3_blk * sp.gp_l3_blk < num_thread) {
        ++sp.mb_l3_blk;
    }
    while (sp.mb_l3_blk * (sp.gp_l3_blk + 1) * im2col_size_per_img < l3_cap_all_core && sp.gp_l3_blk < cp.group && sp.mb_l3_blk * sp.gp_l3_blk < num_thread) {
        ++sp.gp_l3_blk;
    }
}

uint64_t conv2d_im2col_gemm_fp32_sse_executor::cal_temp_buffer_size()
{
    const conv2d_fp32_param &cp = *conv_param_;
    kernel_schedule_param &sp   = schedule_param_;

    const int32_t dst_hw = dst_shape_->GetDim(2) * dst_shape_->GetDim(3);
    const bool is_gemm = cp.is_pointwise() && cp.sparse_level() == 1.0f;

    const uint64_t im2col_size = is_gemm ? 0 : round_up(sp.mb_l3_blk * sp.gp_l3_blk * sp.k_per_gp * dst_hw * sizeof(float), PPL_X86_CACHELINE_BYTES());
    const uint64_t src_trans_size_per_thr = k_l2_blk_max * hw_l2_blk_max * sizeof(float);
    const uint64_t dst_buf_size_per_thr = sp.oc_per_gp * hw_ker_blk_max * sizeof(float);

    return im2col_size + (src_trans_size_per_thr + dst_buf_size_per_thr) * PPL_OMP_MAX_THREADS();
}

ppl::common::RetCode conv2d_im2col_gemm_fp32_sse_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_ || ((conv_param_->fuse_flag & conv_fuse_flag::sum) && !sum_src_shape_)) {
        return ppl::common::RC_INVALID_VALUE;
    }

    init_preproc_param();
    cal_kernel_tunning_param();

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_im2col_gemm_fp32_sse_executor::execute()
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
    const int32_t src_c = src_shape_->GetDim(1);
    const int32_t dst_c = dst_shape_->GetDim(1);

    const int64_t src_g_stride  = int64_t(sp.ic_per_gp) * src_h * src_w;
    const int64_t src_b_stride  = int64_t(src_c) * src_h * src_w;
    const int64_t dst_g_stride  = int64_t(sp.oc_per_gp) * dst_h * dst_w;
    const int64_t dst_b_stride  = int64_t(dst_c) * dst_h * dst_w;
    const int64_t dst_c_stride  = int64_t(dst_h) * dst_w;
    const int64_t flt_g_stride  = int64_t(sp.oc_per_gp) * sp.padded_k;
    const int64_t bias_g_stride = sp.oc_per_gp;
    const int64_t dst_hw = int64_t(dst_h) * dst_w;

    const bool with_sum   = cp.fuse_flag & conv_fuse_flag::sum;
    const bool with_relu  = cp.fuse_flag & conv_fuse_flag::relu;
    const bool with_relu6 = cp.fuse_flag & conv_fuse_flag::relu6;
    const bool is_gemm = cp.is_pointwise() && cp.sparse_level() == 1.0f;

    int64_t sum_src_b_stride = 0;
    if (with_sum) {
        sum_src_b_stride = int64_t(sum_src_shape_->GetDim(1)) * dst_h * dst_w;
    }

    const uint64_t im2col_len = is_gemm ? 0 : (round_up(sp.mb_l3_blk * sp.gp_l3_blk * sp.k_per_gp * dst_hw * sizeof(float), PPL_X86_CACHELINE_BYTES()) / sizeof(float));
    const uint64_t src_trans_len_per_thr = k_l2_blk_max * hw_l2_blk_max;
    const uint64_t dst_buf_len_per_thr = sp.oc_per_gp * hw_ker_blk_max;
    const uint64_t buffer_len_per_thr = src_trans_len_per_thr + dst_buf_len_per_thr;

    float *base_im2col   = (float *)temp_buffer_;
    float *base_buffer = base_im2col + im2col_len;
    for (int64_t gpl3 = 0; gpl3 < cp.group; gpl3 += sp.gp_l3_blk) {
        const int64_t gpl3_eff = min<int64_t>(cp.group - gpl3, sp.gp_l3_blk);
        for (int64_t mbl3 = 0; mbl3 < batch; mbl3 += sp.mb_l3_blk) {
            const int64_t mbl3_eff    = min<int64_t>(batch - mbl3, sp.mb_l3_blk);
            const float *base_src     = src_;
            const float *base_his     = dst_;
            const float *base_flt     = cvt_filter_;
            float *base_dst           = dst_;
            int64_t base_src_b_stride = src_b_stride;
            int64_t base_src_g_stride = src_g_stride;
            int64_t base_his_b_stride = dst_b_stride;
            int64_t base_dst_b_stride = dst_b_stride;
            if (with_sum) {
                base_his          = sum_src_;
                base_his_b_stride = sum_src_b_stride;
            }
            base_src += mbl3 * base_src_b_stride + gpl3 * base_src_g_stride;
            base_his += mbl3 * base_his_b_stride + gpl3 * dst_g_stride;
            base_dst += mbl3 * base_dst_b_stride + gpl3 * dst_g_stride;
            base_flt += gpl3 * flt_g_stride;
            int64_t im2col_b_stride;
            int64_t im2col_g_stride;
            if (is_gemm) {
                base_im2col = const_cast<float*>(base_src);
                im2col_b_stride = cp.group * sp.k_per_gp * dst_hw;
                im2col_g_stride = sp.k_per_gp * dst_hw;
            } else {
                // im2col + reorder_src
                im2col_b_stride = sp.k_per_gp * dst_hw;
                im2col_g_stride = sp.mb_l3_blk * sp.k_per_gp * dst_hw;
#ifdef PPL_USE_X86_OMP_COLLAPSE
                PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
#endif
                for (int64_t g = 0; g < gpl3_eff; ++g) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
                    PRAGMA_OMP_PARALLEL_FOR()
#endif
                    for (int64_t b = 0; b < mbl3_eff; ++b) {
                        for (int64_t ic = 0; ic < sp.ic_per_gp; ++ic) {
                            const float *b_src = base_src + b * base_src_b_stride + g * base_src_g_stride;
                            float *b_im2col = base_im2col + b * im2col_b_stride + g * im2col_g_stride;
                            for (int64_t kh = 0; kh < cp.kernel_h; ++kh) {
                                for (int64_t kw = 0; kw < cp.kernel_w; ++kw) {
                                    const int64_t k = ic * cp.kernel_h * cp.kernel_w + kh * cp.kernel_w + kw;
                                    const int64_t ekh = kh * cp.dilation_h;
                                    const int64_t ekw = kw * cp.dilation_w;
                                    const int64_t oh_beg = max<int64_t>((int64_t)ceilf((cp.pad_h - ekh) / (float)(cp.stride_h)), 0);
                                    const int64_t oh_end = max<int64_t>(oh_beg, min<int64_t>((int64_t)ceilf((src_h + cp.pad_h - ekh) / (float)(cp.stride_h)), dst_h));
                                    const int64_t ow_beg = max<int64_t>((int64_t)ceilf((cp.pad_w - ekw) / (float)(cp.stride_w)), 0);
                                    const int64_t ow_end = max<int64_t>(ow_beg, min<int64_t>((int64_t)ceilf((src_w + cp.pad_w - ekw) / (float)(cp.stride_w)), dst_w));
                                    const int64_t ih_beg = oh_beg * cp.stride_h - cp.pad_h + ekh;
                                    const int64_t iw_beg = ow_beg * cp.stride_w - cp.pad_w + ekw;

                                    const float *l_src = b_src + ic * src_h * src_w;
                                    float *l_im2col = b_im2col + k * dst_h * dst_w;

                                    memset32_sse(l_im2col, 0, oh_beg * dst_w);
                                    if (cp.stride_w == 1) {
                                        for (int64_t oh = oh_beg, ih = ih_beg; oh < oh_end; ++oh, ih += cp.stride_h) {
                                            memset32_sse(l_im2col + oh * dst_w, 0, ow_beg);
                                            memcpy32_sse(l_im2col + oh * dst_w + ow_beg, l_src + ih * src_w + iw_beg, (ow_end - ow_beg));
                                            memset32_sse(l_im2col + oh * dst_w + ow_end, 0, (dst_w - ow_end));
                                        }
                                    } else {
                                        for (int64_t oh = oh_beg, ih = ih_beg; oh < oh_end; ++oh, ih += cp.stride_h) {
                                            memset32_sse(l_im2col + oh * dst_w, 0, ow_beg);
                                            const float *w_src = l_src + ih * src_w + iw_beg;
                                            float *w_im2col = l_im2col + oh * dst_w + ow_beg;
                                            float *w_im2col_end = w_im2col + (ow_end - ow_beg);
                                            for (; w_im2col < w_im2col_end; ++w_im2col) {
                                                *w_im2col = *w_src;
                                                w_src += cp.stride_w;
                                            }
                                            memset32_sse(l_im2col + oh * dst_w + ow_end, 0, (dst_w - ow_end));
                                        }
                                    }
                                    memset32_sse(l_im2col + oh_end * dst_w, 0, (dst_h - oh_end) * dst_w);
                                }
                            }
                        }
                    }
                }
            }
            const int64_t hw_l2_cnt = div_up(dst_hw, sp.hw_l2_blk);
            const int64_t num_thread = PPL_OMP_MAX_THREADS();
            int64_t oc_thr_blk = sp.oc_per_gp;
            if (hw_l2_cnt * gpl3_eff * mbl3_eff < num_thread * 0.8) {
                oc_thr_blk = max<int64_t>(1, sp.oc_per_gp / div_up(num_thread, hw_l2_cnt * gpl3_eff * mbl3_eff));
            }
#ifdef PPL_USE_X86_OMP_COLLAPSE
            PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(4)
#endif
            for (int64_t g = 0; g < gpl3_eff; ++g) {
                for (int64_t b = 0; b < mbl3_eff; ++b) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
                    PRAGMA_OMP_PARALLEL_FOR()
#endif
                    for (int64_t hwl2 = 0; hwl2 < dst_hw; hwl2 += sp.hw_l2_blk) {
                        for (int64_t octhr = 0; octhr < sp.oc_per_gp; octhr += oc_thr_blk) {
                            const int64_t hwl2_eff = min<int64_t>(dst_hw - hwl2, sp.hw_l2_blk);
                            const int64_t hw_body = round(hwl2_eff, hw_ker_blk_max);
                            const int64_t hw_tail = hwl2_eff - hw_body;
                            const int64_t hw_tregb = div_up(hw_tail, hw_regb_elts);
                            const int64_t octhr_eff = min<int64_t>(sp.oc_per_gp - octhr, oc_thr_blk);
                            const int64_t dst_buf_c_stride = hw_ker_blk_max;

                            int64_t kernel_param[conv_gemm_kernel_fp32_sse::param_def::length];
                            array_param_helper kp(kernel_param);
                            conv_gemm_kernel_fp32_sse ker(kernel_param);
                            kp.pick<int64_t>(conv_gemm_kernel_fp32_sse::param_def::flags_idx) = conv_gemm_kernel_fp32_sse::flag::load_h;
                            kp.pick<int64_t>(conv_gemm_kernel_fp32_sse::param_def::m_idx) = octhr_eff;

                            float *thr_src_trans = base_buffer + buffer_len_per_thr * PPL_OMP_THREAD_ID();
                            float *thr_dst_buf = thr_src_trans + src_trans_len_per_thr;
                            const float *thr_flt = base_flt + g * flt_g_stride;
                            const float *thr_bias = cvt_bias_ + (gpl3 + g) * bias_g_stride + octhr;
                            const float *thr_im2col = base_im2col + b * im2col_b_stride + g * im2col_g_stride + hwl2;
                            float *thr_dst = base_dst + b * base_dst_b_stride + g * dst_g_stride + octhr * dst_hw + hwl2;
                            // bias, eltwise
                            {
                                float *l_dst_buf = thr_dst_buf;
                                float *l_dst = thr_dst;
                                if (with_sum) {
                                    const float *l_his = base_his + b * base_his_b_stride + g * dst_g_stride + octhr * dst_hw + hwl2;
                                    for (int64_t oc = 0; oc < octhr_eff; ++oc) {
                                        if (hw_body) {
                                            __m128 xmm_bias = _mm_set1_ps(thr_bias[oc]);
                                            for (int64_t hw = 0; hw < hw_body; hw += hw_regb_elts) {
                                                _mm_storeu_ps(l_dst + hw + 0 * hw_reg_elts, _mm_add_ps(_mm_loadu_ps(l_his + hw + 0 * hw_reg_elts), xmm_bias));
                                                _mm_storeu_ps(l_dst + hw + 1 * hw_reg_elts, _mm_add_ps(_mm_loadu_ps(l_his + hw + 1 * hw_reg_elts), xmm_bias));
                                            }
                                            l_dst += dst_c_stride;
                                        }
                                        if (hw_tail) {
                                            for (int64_t hw = hw_body; hw < hwl2_eff; ++hw) {
                                                l_dst_buf[hw - hw_body] = l_his[hw] + thr_bias[oc];
                                            }
                                            l_dst_buf += dst_buf_c_stride;
                                        }
                                        l_his += dst_c_stride;
                                    }
                                } else {
                                    for (int64_t oc = 0; oc < octhr_eff; ++oc) {
                                        if (hw_body) {
                                            memset32_sse(l_dst, (*(int32_t*)&thr_bias[oc]), hw_body);
                                            l_dst += dst_c_stride;
                                        }
                                        if (hw_tail) {
                                            memset32_sse(l_dst_buf, (*(int32_t*)&thr_bias[oc]), hwl2_eff - hw_body);
                                            l_dst_buf += dst_buf_c_stride;
                                        }
                                    }
                                }
                            }
                            // gemm
                            for (int64_t kl2 = 0; kl2 < sp.k_per_gp; kl2 += k_l2_blk_max) {
                                const int64_t kl2_eff = min<int64_t>(sp.k_per_gp - kl2, k_l2_blk_max);
                                const bool is_last_k  = kl2 + k_l2_blk_max >= sp.k_per_gp;
                                if (is_last_k) {
                                    if (with_relu) {
                                        kp.pick<int64_t>(conv_gemm_kernel_fp32_sse::param_def::flags_idx) |= conv_gemm_kernel_fp32_sse::flag::relu;
                                    } else if (with_relu6) {
                                        kp.pick<int64_t>(conv_gemm_kernel_fp32_sse::param_def::flags_idx) |= conv_gemm_kernel_fp32_sse::flag::relu6;
                                    }
                                }
                                // pack col, Nk8n
                                {
                                    int64_t hw = 0;
                                    for (; hw < hwl2_eff; hw += hw_regb_elts) {
                                        const float *l_im2col = thr_im2col + kl2 * dst_hw + hw;
                                        float *l_src_trans = thr_src_trans + hw * kl2_eff;
                                        for (int64_t k = 0; k < kl2_eff; ++k) {
                                            _mm_storeu_ps(l_src_trans + 0 * hw_reg_elts, _mm_loadu_ps(l_im2col + 0 * hw_reg_elts));
                                            _mm_storeu_ps(l_src_trans + 1 * hw_reg_elts, _mm_loadu_ps(l_im2col + 1 * hw_reg_elts));
                                            l_im2col += dst_hw;
                                            l_src_trans += hw_regb_elts;
                                        }
                                    }
                                    if (hw < hwl2_eff) {
                                        const int64_t hw_rb_tail = hwl2_eff - hw;
                                        const float *l_im2col = thr_im2col + kl2 * dst_hw + hw;
                                        float *l_src_trans = thr_src_trans + hw * kl2_eff;
                                        for (int64_t k = 0; k < kl2_eff; ++k) {
                                            const float *d_im2col = l_im2col;
                                            float *d_src_trans = l_src_trans;
                                            if (hw_rb_tail & 4) {
                                                _mm_storeu_ps(d_src_trans, _mm_loadu_ps(d_im2col));
                                                d_src_trans += 4;
                                                d_im2col += 4;
                                            }
                                            if (hw_rb_tail & 2) {
                                                d_src_trans[0] = d_im2col[0];
                                                d_src_trans[1] = d_im2col[1];
                                                d_src_trans += 2;
                                                d_im2col += 2;
                                            }
                                            if (hw_rb_tail & 1) {
                                                d_src_trans[0] = d_im2col[0];
                                            }
                                            l_im2col += dst_hw;
                                            l_src_trans += hw_regb_elts;
                                        }
                                    }
                                }
                                kp.pick<int64_t>(conv_gemm_kernel_fp32_sse::param_def::k_idx) = kl2_eff;
                                kp.pick<int64_t>(conv_gemm_kernel_fp32_sse::param_def::lda_idx) = kl2_eff;
                                kp.pick<int64_t>(conv_gemm_kernel_fp32_sse::param_def::ldpacked_b_idx) = kl2_eff * hw_regb_elts;
                                kp.pick<const float*>(conv_gemm_kernel_fp32_sse::param_def::a_ptr_idx) = thr_flt + kl2 * sp.oc_per_gp + octhr * kl2_eff;
                                if (hw_body) {
                                    kp.pick<int64_t>(conv_gemm_kernel_fp32_sse::param_def::ldc_idx) = dst_c_stride;
                                    kp.pick<int64_t>(conv_gemm_kernel_fp32_sse::param_def::ldh_idx) = dst_c_stride;
                                    for (int64_t hw = 0; hw < hw_body; hw += hw_ker_blk_max) {
                                        kp.pick<const float*>(conv_gemm_kernel_fp32_sse::param_def::packed_b_ptr_idx) = thr_src_trans + hw * kl2_eff;
                                        kp.pick<float*>(conv_gemm_kernel_fp32_sse::param_def::c_ptr_idx) = thr_dst + hw;
                                        kp.pick<float*>(conv_gemm_kernel_fp32_sse::param_def::h_ptr_idx) = thr_dst + hw;
                                        ker.execute(conv_gemm_kernel_fp32_sse::config::max_n_regbs);
                                    }
                                }
                                if (hw_tail) {
                                    kp.pick<int64_t>(conv_gemm_kernel_fp32_sse::param_def::ldc_idx) = dst_buf_c_stride;
                                    kp.pick<int64_t>(conv_gemm_kernel_fp32_sse::param_def::ldh_idx) = dst_buf_c_stride;
                                    kp.pick<const float*>(conv_gemm_kernel_fp32_sse::param_def::packed_b_ptr_idx) = thr_src_trans + hw_body * kl2_eff;
                                    kp.pick<float*>(conv_gemm_kernel_fp32_sse::param_def::c_ptr_idx) = thr_dst_buf;
                                    kp.pick<float*>(conv_gemm_kernel_fp32_sse::param_def::h_ptr_idx) = thr_dst_buf;
                                    ker.execute(hw_tregb);
                                }
                            }
                            // store dst
                            if (hw_tail) {
                                float *l_dst_buf = thr_dst_buf;
                                float *l_dst = base_dst + b * base_dst_b_stride + g * dst_g_stride + octhr * dst_hw + hwl2;
                                for (int64_t oc = 0; oc < octhr_eff; ++oc) {
                                    memcpy32_sse(l_dst + hw_body, l_dst_buf, hwl2_eff - hw_body);
                                    l_dst_buf += dst_buf_c_stride;
                                    l_dst += dst_c_stride;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_im2col_gemm_fp32_sse_manager::gen_cvt_weights(const float *filter, const float *bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    cvt_bias_size_ = param_.num_output;
    cvt_bias_      = (float *)allocator_->Alloc(cvt_bias_size_ * sizeof(float));
    if (cvt_bias_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    memcpy(cvt_bias_, bias, param_.num_output * sizeof(float));

    // M = OC
    // K = ic * kh * kw
    // filter: MK -> KMBk
    const int64_t ic_per_gp = param_.channels / param_.group;
    const int64_t oc_per_gp = param_.num_output / param_.group;
    const int64_t k_per_gp = ic_per_gp * param_.kernel_h * param_.kernel_w;

    cvt_filter_size_ = param_.group * oc_per_gp * k_per_gp;
    cvt_filter_      = (float *)allocator_->Alloc(cvt_filter_size_ * sizeof(float));
    if (cvt_filter_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    for (int64_t g = 0; g < param_.group; ++g) {
        for (int64_t k = 0; k < k_per_gp; k += k_l2_blk_max) {
            const int64_t k_eff = min<int64_t>(k_per_gp - k, k_l2_blk_max);
            const float *l_flt = filter + g * oc_per_gp * k_per_gp + k;
            float *l_cvt_flt = cvt_filter_ + g * oc_per_gp * k_per_gp + k * oc_per_gp;
            for (int64_t oc = 0; oc < oc_per_gp; ++oc) {
                memcpy(l_cvt_flt, l_flt, k_eff * sizeof(float));
                l_flt += k_per_gp;
                l_cvt_flt += k_eff;
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

bool conv2d_im2col_gemm_fp32_sse_manager::is_supported()
{
    return true;
}

conv2d_fp32_executor *conv2d_im2col_gemm_fp32_sse_manager::gen_executor()
{
    return new conv2d_im2col_gemm_fp32_sse_executor(&param_, cvt_filter_, cvt_bias_);
}

}}}; // namespace ppl::kernel::x86
