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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DEPTHWISE_SSE_CONV2D_N8CX_DEPTHWISE_BLK1X7_KERNEL_FP32_SSE_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DEPTHWISE_SSE_CONV2D_N8CX_DEPTHWISE_BLK1X7_KERNEL_FP32_SSE_H_

#include <nmmintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/sse/conv2d_n8cx_depthwise_kernel_fp32_sse.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store, int32_t spec_stride_w, int32_t w_len>
void conv2d_n8cx_depthwise_fp32_sse_blk1x7_kernel(
    const int64_t *shar_param,
    int64_t *priv_param)
{
#define KW_COMPUTE_STEP() do {\
    xmm14 = _mm_loadu_ps(k_flt + 0 * CH_RF_BLK());\
    if (w_len > 0) {\
        xmm15 = _mm_loadu_ps(k_src + 0 * src_sw_stride + 0 * CH_RF_BLK());\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm0 = _mm_add_ps(xmm0, xmm15);\
    }\
    if (w_len > 1) {\
        xmm15 = _mm_loadu_ps(k_src + 1 * src_sw_stride + 0 * CH_RF_BLK());\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm2 = _mm_add_ps(xmm2, xmm15);\
    }\
    if (w_len > 2) {\
        xmm15 = _mm_loadu_ps(k_src + 2 * src_sw_stride + 0 * CH_RF_BLK());\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm4 = _mm_add_ps(xmm4, xmm15);\
    }\
    if (w_len > 3) {\
        xmm15 = _mm_loadu_ps(k_src + 3 * src_sw_stride + 0 * CH_RF_BLK());\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm6 = _mm_add_ps(xmm6, xmm15);\
    }\
    if (w_len > 4) {\
        xmm15 = _mm_loadu_ps(k_src + 4 * src_sw_stride + 0 * CH_RF_BLK());\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm8 = _mm_add_ps(xmm8, xmm15);\
    }\
    if (w_len > 5) {\
        xmm15 = _mm_loadu_ps(k_src + 5 * src_sw_stride + 0 * CH_RF_BLK());\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm10 = _mm_add_ps(xmm10, xmm15);\
    }\
    if (w_len > 6) {\
        xmm15 = _mm_loadu_ps(k_src + 6 * src_sw_stride + 0 * CH_RF_BLK());\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm12 = _mm_add_ps(xmm12, xmm15);\
    }\
    xmm14 = _mm_loadu_ps(k_flt + 1 * CH_RF_BLK());\
    if (w_len > 0) {\
        xmm15 = _mm_loadu_ps(k_src + 0 * src_sw_stride + 1 * CH_RF_BLK());\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm1 = _mm_add_ps(xmm1, xmm15);\
    }\
    if (w_len > 1) {\
        xmm15 = _mm_loadu_ps(k_src + 1 * src_sw_stride + 1 * CH_RF_BLK());\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm3 = _mm_add_ps(xmm3, xmm15);\
    }\
    if (w_len > 2) {\
        xmm15 = _mm_loadu_ps(k_src + 2 * src_sw_stride + 1 * CH_RF_BLK());\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm5 = _mm_add_ps(xmm5, xmm15);\
    }\
    if (w_len > 3) {\
        xmm15 = _mm_loadu_ps(k_src + 3 * src_sw_stride + 1 * CH_RF_BLK());\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm7 = _mm_add_ps(xmm7, xmm15);\
    }\
    if (w_len > 4) {\
        xmm15 = _mm_loadu_ps(k_src + 4 * src_sw_stride + 1 * CH_RF_BLK());\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm9 = _mm_add_ps(xmm9, xmm15);\
    }\
    if (w_len > 5) {\
        xmm15 = _mm_loadu_ps(k_src + 5 * src_sw_stride + 1 * CH_RF_BLK());\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm11 = _mm_add_ps(xmm11, xmm15);\
    }\
    if (w_len > 6) {\
        xmm15 = _mm_loadu_ps(k_src + 6 * src_sw_stride + 1 * CH_RF_BLK());\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm13 = _mm_add_ps(xmm13, xmm15);\
    }\
    k_flt += CH_DT_BLK();\
    k_src += src_dw_stride;\
} while (0)

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
    __m128 xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15;

    const int64_t src_sw_stride = spec_stride_w ? spec_stride_w * CH_DT_BLK() : shar_param[SRC_SW_STRIDE_IDX()];
    const int64_t src_dh_stride = shar_param[SRC_DH_STRIDE_IDX()];
    const int64_t src_dw_stride = shar_param[SRC_DW_STRIDE_IDX()];
    const int64_t kernel_flags  = shar_param[FLAGS_IDX()];
    const int64_t kernel_w      = shar_param[KW_IDX()];
    const int64_t src_kh_stride = src_dh_stride - kernel_w * src_dw_stride;

    const int64_t kh_start = priv_param[KH_START_IDX()];
    const int64_t kh_end   = priv_param[KH_END_IDX()];

    const float *src = PICK_PARAM(const float*, priv_param, SRC_IDX());
    const float *sum = PICK_PARAM(const float*, priv_param, SUM_SRC_IDX());
    float *dst       = PICK_PARAM(float*, priv_param, DST_IDX());
    int64_t ow       = priv_param[OW_IDX()];
    do {
        const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
        if (w_len > 0) {
            xmm0 = _mm_loadu_ps(bias + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
            xmm1 = _mm_loadu_ps(bias + 0 * CH_DT_BLK() + 1 * CH_RF_BLK());
        }
        if (w_len > 1) {
            xmm2 = xmm0;
            xmm3 = xmm1;
        }
        if (w_len > 2) {
            xmm4 = xmm0;
            xmm5 = xmm1;
        }
        if (w_len > 3) {
            xmm6 = xmm0;
            xmm7 = xmm1;
        }
        if (w_len > 4) {
            xmm8 = xmm0;
            xmm9 = xmm1;
        }
        if (w_len > 5) {
            xmm10 = xmm0;
            xmm11 = xmm1;
        }
        if (w_len > 6) {
            xmm12 = xmm0;
            xmm13 = xmm1;
        }

        const float *k_src = src + kh_start * src_dh_stride;
        const float *k_flt  = PICK_PARAM(const float*, priv_param, FLT_IDX()) + kh_start * kernel_w * CH_DT_BLK();
        for (int32_t kh = kh_start; kh < kh_end; ++kh) {
            for (int32_t kw = 0; kw < kernel_w; ++kw) {
                KW_COMPUTE_STEP();
            }
            k_src += src_kh_stride;
        }
        
        if (kernel_flags & KERNEL_FLAG_SUM()) {
            if (w_len > 0) {
                xmm14 = _mm_loadu_ps(sum + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm15 = _mm_loadu_ps(sum + 0 * CH_DT_BLK() + 1 * CH_RF_BLK());
                xmm0 = _mm_add_ps(xmm0, xmm14);
                xmm1 = _mm_add_ps(xmm1, xmm15);
            }
            if (w_len > 1) {
                xmm14 = _mm_loadu_ps(sum + 1 * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm15 = _mm_loadu_ps(sum + 1 * CH_DT_BLK() + 1 * CH_RF_BLK());
                xmm2 = _mm_add_ps(xmm2, xmm14);
                xmm3 = _mm_add_ps(xmm3, xmm15);
            }
            if (w_len > 2) {
                xmm14 = _mm_loadu_ps(sum + 2 * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm15 = _mm_loadu_ps(sum + 2 * CH_DT_BLK() + 1 * CH_RF_BLK());
                xmm4 = _mm_add_ps(xmm4, xmm14);
                xmm5 = _mm_add_ps(xmm5, xmm15);
            }
            if (w_len > 3) {
                xmm14 = _mm_loadu_ps(sum + 3 * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm15 = _mm_loadu_ps(sum + 3 * CH_DT_BLK() + 1 * CH_RF_BLK());
                xmm6 = _mm_add_ps(xmm6, xmm14);
                xmm7 = _mm_add_ps(xmm7, xmm15);
            }
            if (w_len > 4) {
                xmm14 = _mm_loadu_ps(sum + 4 * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm15 = _mm_loadu_ps(sum + 4 * CH_DT_BLK() + 1 * CH_RF_BLK());
                xmm8 = _mm_add_ps(xmm8, xmm14);
                xmm9 = _mm_add_ps(xmm9, xmm15);
            }
            if (w_len > 5) {
                xmm14 = _mm_loadu_ps(sum + 5 * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm15 = _mm_loadu_ps(sum + 5 * CH_DT_BLK() + 1 * CH_RF_BLK());
                xmm10 = _mm_add_ps(xmm10, xmm14);
                xmm11 = _mm_add_ps(xmm11, xmm15);
            }
            if (w_len > 6) {
                xmm14 = _mm_loadu_ps(sum + 6 * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm15 = _mm_loadu_ps(sum + 6 * CH_DT_BLK() + 1 * CH_RF_BLK());
                xmm12 = _mm_add_ps(xmm12, xmm14);
                xmm13 = _mm_add_ps(xmm13, xmm15);
            }
        }
        if (kernel_flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
            xmm14 = _mm_setzero_ps();
            if (w_len > 0) {
                xmm0 = _mm_max_ps(xmm0, xmm14);
                xmm1 = _mm_max_ps(xmm1, xmm14);
            }
            if (w_len > 1) {
                xmm2 = _mm_max_ps(xmm2, xmm14);
                xmm3 = _mm_max_ps(xmm3, xmm14);
            }
            if (w_len > 2) {
                xmm4 = _mm_max_ps(xmm4, xmm14);
                xmm5 = _mm_max_ps(xmm5, xmm14);
            }
            if (w_len > 3) {
                xmm6 = _mm_max_ps(xmm6, xmm14);
                xmm7 = _mm_max_ps(xmm7, xmm14);
            }
            if (w_len > 4) {
                xmm8 = _mm_max_ps(xmm8, xmm14);
                xmm9 = _mm_max_ps(xmm9, xmm14);
            }
            if (w_len > 5) {
                xmm10 = _mm_max_ps(xmm10, xmm14);
                xmm11 = _mm_max_ps(xmm11, xmm14);
            }
            if (w_len > 6) {
                xmm12 = _mm_max_ps(xmm12, xmm14);
                xmm13 = _mm_max_ps(xmm13, xmm14);
            }
        }
        if (kernel_flags & KERNEL_FLAG_RELU6()) {
            xmm15 = _mm_set1_ps(6.0f);
            if (w_len > 0) {
                xmm0 = _mm_min_ps(xmm0, xmm15);
                xmm1 = _mm_min_ps(xmm1, xmm15);
            }
            if (w_len > 1) {
                xmm2 = _mm_min_ps(xmm2, xmm15);
                xmm3 = _mm_min_ps(xmm3, xmm15);
            }
            if (w_len > 2) {
                xmm4 = _mm_min_ps(xmm4, xmm15);
                xmm5 = _mm_min_ps(xmm5, xmm15);
            }
            if (w_len > 3) {
                xmm6 = _mm_min_ps(xmm6, xmm15);
                xmm7 = _mm_min_ps(xmm7, xmm15);
            }
            if (w_len > 4) {
                xmm8 = _mm_min_ps(xmm8, xmm15);
                xmm9 = _mm_min_ps(xmm9, xmm15);
            }
            if (w_len > 5) {
                xmm10 = _mm_min_ps(xmm10, xmm15);
                xmm11 = _mm_min_ps(xmm11, xmm15);
            }
            if (w_len > 6) {
                xmm12 = _mm_min_ps(xmm12, xmm15);
                xmm13 = _mm_min_ps(xmm13, xmm15);
            }
        }
        if (nt_store) {
            if (w_len > 0) {
                _mm_stream_ps(dst + 0 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm0);
                _mm_stream_ps(dst + 0 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm1);
            }
            if (w_len > 1) {
                _mm_stream_ps(dst + 1 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm2);
                _mm_stream_ps(dst + 1 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm3);
            }
            if (w_len > 2) {
                _mm_stream_ps(dst + 2 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm4);
                _mm_stream_ps(dst + 2 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm5);
            }
            if (w_len > 3) {
                _mm_stream_ps(dst + 3 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm6);
                _mm_stream_ps(dst + 3 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm7);
            }
            if (w_len > 4) {
                _mm_stream_ps(dst + 4 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm8);
                _mm_stream_ps(dst + 4 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm9);
            }
            if (w_len > 5) {
                _mm_stream_ps(dst + 5 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm10);
                _mm_stream_ps(dst + 5 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm11);
            }
            if (w_len > 6) {
                _mm_stream_ps(dst + 6 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm12);
                _mm_stream_ps(dst + 6 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm13);
            }
        } else {
            if (w_len > 0) {
                _mm_storeu_ps(dst + 0 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm0);
                _mm_storeu_ps(dst + 0 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm1);
            }
            if (w_len > 1) {
                _mm_storeu_ps(dst + 1 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm2);
                _mm_storeu_ps(dst + 1 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm3);
            }
            if (w_len > 2) {
                _mm_storeu_ps(dst + 2 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm4);
                _mm_storeu_ps(dst + 2 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm5);
            }
            if (w_len > 3) {
                _mm_storeu_ps(dst + 3 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm6);
                _mm_storeu_ps(dst + 3 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm7);
            }
            if (w_len > 4) {
                _mm_storeu_ps(dst + 4 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm8);
                _mm_storeu_ps(dst + 4 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm9);
            }
            if (w_len > 5) {
                _mm_storeu_ps(dst + 5 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm10);
                _mm_storeu_ps(dst + 5 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm11);
            }
            if (w_len > 6) {
                _mm_storeu_ps(dst + 6 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm12);
                _mm_storeu_ps(dst + 6 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm13);
            }
        }
        src += w_len * src_sw_stride;
        sum += w_len * CH_DT_BLK();
        dst += w_len * CH_DT_BLK();
        ow -= w_len;
    } while (ow > 0);
    PICK_PARAM(const float *, priv_param, SRC_IDX()) = src;
    PICK_PARAM(const float *, priv_param, SUM_SRC_IDX()) = sum;
    PICK_PARAM(float *, priv_param, DST_IDX()) = dst;
#undef KW_COMPUTE_STEP
}

}}};

#endif
