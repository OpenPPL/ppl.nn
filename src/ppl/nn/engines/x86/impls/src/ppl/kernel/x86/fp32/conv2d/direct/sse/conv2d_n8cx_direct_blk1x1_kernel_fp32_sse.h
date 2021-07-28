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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_SSE_CONV2D_N8CX_DIRECT_BLK1X1_KERNEL_FP32_SSE_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_SSE_CONV2D_N8CX_DIRECT_BLK1X1_KERNEL_FP32_SSE_H_

#include <nmmintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/direct/sse/conv2d_n8cx_direct_kernel_fp32_sse.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store, int32_t oc_len>
void conv2d_n8cx_direct_fp32_sse_blk1x1_kernel(
    const int64_t *shar_param,
    int64_t *priv_param)
{
#define IC_COMPUTE_STEP(IC) do {\
    xmm4 = _mm_set1_ps(ic_src[(IC)]);\
    if (oc_len > 0 * CH_DT_BLK()) {\
        xmm5 = _mm_loadu_ps(ic_flt + 0 * flt_ocb_stride + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
        xmm5 = _mm_mul_ps(xmm5, xmm4);\
        xmm0 = _mm_add_ps(xmm0, xmm5);\
        xmm6 = _mm_loadu_ps(ic_flt + 0 * flt_ocb_stride + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
        xmm6 = _mm_mul_ps(xmm6, xmm4);\
        xmm1 = _mm_add_ps(xmm1, xmm6);\
    }\
    if (oc_len > 1 * CH_DT_BLK()) {\
        xmm7 = _mm_loadu_ps(ic_flt + 1 * flt_ocb_stride + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
        xmm7 = _mm_mul_ps(xmm7, xmm4);\
        xmm2 = _mm_add_ps(xmm2, xmm7);\
        xmm8 = _mm_loadu_ps(ic_flt + 1 * flt_ocb_stride + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
        xmm8 = _mm_mul_ps(xmm8, xmm4);\
        xmm3 = _mm_add_ps(xmm3, xmm8);\
    }\
} while (0)

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
    __m128 xmm8, xmm9, xmm10; // xmm11, xmm12, xmm13, xmm14, xmm15;

    const uint64_t kernel_flags = PICK_PARAM(const uint64_t, shar_param, FLAGS_IDX());
    if (kernel_flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
        xmm9 = _mm_setzero_ps();
    }
    if (kernel_flags & KERNEL_FLAG_RELU6()) {
        xmm10 = _mm_set1_ps(6.0f);
    }

    if (kernel_flags & KERNEL_FLAG_LD_BIAS()) {
        const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
        if (oc_len > 0 * CH_DT_BLK()) {
            xmm0 = _mm_loadu_ps(bias + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
            xmm1 = _mm_loadu_ps(bias + 0 * CH_DT_BLK() + 1 * CH_RF_BLK());
        }
        if (oc_len > 1 * CH_DT_BLK()) {
            xmm2 = _mm_loadu_ps(bias + 1 * CH_DT_BLK() + 0 * CH_RF_BLK());
            xmm3 = _mm_loadu_ps(bias + 1 * CH_DT_BLK() + 1 * CH_RF_BLK());
        }
    } else {
        const float* his = PICK_PARAM(const float*, priv_param, HIS_IDX());
        const int64_t his_ocb_stride = shar_param[HIS_OCB_STRIDE_IDX()];
        if (oc_len > 0 * CH_DT_BLK()) {
            xmm0 = _mm_loadu_ps(his + 0 * CH_RF_BLK());
            xmm1 = _mm_loadu_ps(his + 1 * CH_RF_BLK());
        }
        if (oc_len > 1 * CH_DT_BLK()) {
            his += his_ocb_stride;
            xmm2 = _mm_loadu_ps(his + 0 * CH_RF_BLK());
            xmm3 = _mm_loadu_ps(his + 1 * CH_RF_BLK());
        }
    }

    if (kernel_flags & KERNEL_FLAG_AD_BIAS()) {
        const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
        if (oc_len > 0 * CH_DT_BLK()) {
            xmm4 = _mm_loadu_ps(bias + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
            xmm0 = _mm_add_ps(xmm4, xmm0);
            xmm5 = _mm_loadu_ps(bias + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
            xmm1 = _mm_add_ps(xmm5, xmm1);
        }
        if (oc_len > 1 * CH_DT_BLK()) {
            xmm6 = _mm_loadu_ps(bias + 1 * CH_DT_BLK() + 0 * CH_RF_BLK());
            xmm2 = _mm_add_ps(xmm6, xmm2);
            xmm7 = _mm_loadu_ps(bias + 1 * CH_DT_BLK() + 1 * CH_RF_BLK());
            xmm3 = _mm_add_ps(xmm7, xmm3);
        }
    }

    const int64_t kernel_h = shar_param[KH_IDX()];
    const int64_t kernel_w = shar_param[KW_IDX()];
    const int64_t src_icb_stride = shar_param[SRC_ICB_STRIDE_IDX()];
    const int64_t src_dh_stride = shar_param[SRC_DH_STRIDE_IDX()];
    const int64_t src_dw_stride = shar_param[SRC_DW_STRIDE_IDX()];
    const int64_t src_sw_stride = shar_param[SRC_SW_STRIDE_IDX()];
    const int64_t flt_ocb_stride = shar_param[FLT_OCB_STRIDE_IDX()];
    const int64_t kh_start = priv_param[KH_START_IDX()];
    const int64_t kh_end = priv_param[KH_END_IDX()];
    const int64_t kw_start = priv_param[KW_START_IDX()];
    const int64_t kw_end = priv_param[KW_END_IDX()];

    const float *icb_src = PICK_PARAM(const float*, priv_param, SRC_IDX()) + kh_start * src_dh_stride;
    const float *icb_flt = PICK_PARAM(const float*, priv_param, FLT_IDX()) + kh_start * kernel_w * CH_DT_BLK() * CH_DT_BLK();
    int64_t channels = shar_param[CHANNELS_IDX()];
    do {
        const float *kh_src = icb_src;
        const float *kh_flt = icb_flt;
        for (int64_t kh = kh_start; kh < kh_end; ++kh) {
            const float *kw_src = kh_src + kw_start * src_dw_stride;
            const float *kw_flt = kh_flt + kw_start * CH_DT_BLK() * CH_DT_BLK();
            for (int64_t kw = kw_start; kw < kw_end; ++kw) {
                if (channels >= CH_DT_BLK()) {
                    const float *ic_src = kw_src;
                    const float *ic_flt = kw_flt;
                    for (int64_t ic = 0; ic < CH_DT_BLK(); ++ic) {
                        IC_COMPUTE_STEP(0);
                        ic_src += 1;
                        ic_flt += CH_DT_BLK();
                    }
                } else {
                    const float *ic_src = kw_src;
                    const float *ic_flt = kw_flt;
                    for (int64_t ic = 0; ic < channels; ++ic) {
                        IC_COMPUTE_STEP(0);
                        ic_src += 1;
                        ic_flt += CH_DT_BLK();
                    }
                }
                kw_flt += CH_DT_BLK() * CH_DT_BLK();
                kw_src += src_dw_stride;
            }
            kh_flt += kernel_w * CH_DT_BLK() * CH_DT_BLK();
            kh_src += src_dh_stride;
        }
        icb_flt += kernel_h * kernel_w * CH_DT_BLK() * CH_DT_BLK();
        icb_src += src_icb_stride;
        channels -= CH_DT_BLK();
    } while (channels > 0);
    
    if (kernel_flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
        if (oc_len > 0 * CH_DT_BLK()) {
            xmm0 = _mm_max_ps(xmm0, xmm9);
            xmm1 = _mm_max_ps(xmm1, xmm9);
        }
        if (oc_len > 1 * CH_DT_BLK()) {
            xmm2 = _mm_max_ps(xmm2, xmm9);
            xmm3 = _mm_max_ps(xmm3, xmm9);
        }
    }
    if (kernel_flags & KERNEL_FLAG_RELU6()) {
        if (oc_len > 0 * CH_DT_BLK()) {
            xmm0 = _mm_min_ps(xmm0, xmm10);
            xmm1 = _mm_min_ps(xmm1, xmm10);
        }
        if (oc_len > 1 * CH_DT_BLK()) {
            xmm2 = _mm_min_ps(xmm2, xmm10);
            xmm3 = _mm_min_ps(xmm3, xmm10);
        }
    }

    if (nt_store) {
        float* dst = PICK_PARAM(float*, priv_param, DST_IDX());
        const int64_t dst_ocb_stride = shar_param[DST_OCB_STRIDE_IDX()];
        if (oc_len > 0 * CH_DT_BLK()) {
            _mm_stream_ps(dst + 0 * CH_RF_BLK(), xmm0);
            _mm_stream_ps(dst + 1 * CH_RF_BLK(), xmm1);
        }
        if (oc_len > 1 * CH_DT_BLK()) {
            dst += dst_ocb_stride;
            _mm_stream_ps(dst + 0 * CH_RF_BLK(), xmm2);
            _mm_stream_ps(dst + 1 * CH_RF_BLK(), xmm3);
        }
    } else {
        float* dst = PICK_PARAM(float*, priv_param, DST_IDX());
        const int64_t dst_ocb_stride = shar_param[DST_OCB_STRIDE_IDX()];
        if (oc_len > 0 * CH_DT_BLK()) {
            _mm_storeu_ps(dst + 0 * CH_RF_BLK(), xmm0);
            _mm_storeu_ps(dst + 1 * CH_RF_BLK(), xmm1);
        }
        if (oc_len > 1 * CH_DT_BLK()) {
            dst += dst_ocb_stride;
            _mm_storeu_ps(dst + 0 * CH_RF_BLK(), xmm2);
            _mm_storeu_ps(dst + 1 * CH_RF_BLK(), xmm3);
        }
    }
    PICK_PARAM(const float *, priv_param, SRC_IDX()) += src_sw_stride;
    PICK_PARAM(const float *, priv_param, HIS_IDX()) += CH_DT_BLK();
    PICK_PARAM(float *, priv_param, DST_IDX()) += CH_DT_BLK();
#undef IC_COMPUTE_STEP
}

}}};

#endif
