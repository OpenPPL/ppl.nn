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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DEPTHWISE_SSE_CONV2D_N8CX_DEPTHWISE_BLK1X1_KERNEL_FP32_SSE_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DEPTHWISE_SSE_CONV2D_N8CX_DEPTHWISE_BLK1X1_KERNEL_FP32_SSE_H_

#include <nmmintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/depthwise/sse/conv2d_n8cx_depthwise_kernel_fp32_sse.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store>
void conv2d_n8cx_depthwise_fp32_sse_blk1x1_kernel(
    const int64_t *shar_param,
    int64_t *priv_param)
{
    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;

    const int64_t src_dh_stride = shar_param[SRC_DH_STRIDE_IDX()];
    const int64_t src_dw_stride = shar_param[SRC_DW_STRIDE_IDX()];
    const int64_t src_sw_stride = shar_param[SRC_SW_STRIDE_IDX()];
    const int64_t kernel_flags  = shar_param[FLAGS_IDX()];
    const int64_t kernel_w      = shar_param[KW_IDX()];

    const int64_t kh_start = priv_param[KH_START_IDX()];
    const int64_t kh_end   = priv_param[KH_END_IDX()];
    const int64_t kw_start = priv_param[KW_START_IDX()];
    const int64_t kw_end   = priv_param[KW_END_IDX()];

    const float *bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
    xmm0 = _mm_loadu_ps(bias + 0 * CH_RF_BLK());
    xmm1 = _mm_loadu_ps(bias + 1 * CH_RF_BLK());

    const float *k_src = PICK_PARAM(const float*, priv_param, SRC_IDX()) + kh_start * src_dh_stride;
    const float *k_flt = PICK_PARAM(const float*, priv_param, FLT_IDX()) + kh_start * kernel_w * CH_DT_BLK();
    if (src_dw_stride == CH_DT_BLK()) {
        for (int32_t kh = kh_start; kh < kh_end; ++kh) {
            for (int32_t kw = kw_start; kw < kw_end; ++kw) {
                xmm2 = _mm_loadu_ps(k_flt + kw * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm3 = _mm_loadu_ps(k_flt + kw * CH_DT_BLK() + 1 * CH_RF_BLK());
                xmm4 = _mm_loadu_ps(k_src + kw * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm5 = _mm_loadu_ps(k_src + kw * CH_DT_BLK() + 1 * CH_RF_BLK());
                xmm2 = _mm_mul_ps(xmm2, xmm4);
                xmm3 = _mm_mul_ps(xmm3, xmm5);
                xmm0 = _mm_add_ps(xmm0, xmm2);
                xmm1 = _mm_add_ps(xmm1, xmm3);
            }
            k_flt += kernel_w * CH_DT_BLK();
            k_src += src_dh_stride;
        }
    } else {
        for (int32_t kh = kh_start; kh < kh_end; ++kh) {
            for (int32_t kw = kw_start; kw < kw_end; ++kw) {
                xmm2 = _mm_loadu_ps(k_flt + kw * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm3 = _mm_loadu_ps(k_flt + kw * CH_DT_BLK() + 1 * CH_RF_BLK());
                xmm4 = _mm_loadu_ps(k_src + kw * src_dw_stride + 0 * CH_RF_BLK());
                xmm5 = _mm_loadu_ps(k_src + kw * src_dw_stride + 1 * CH_RF_BLK());
                xmm2 = _mm_mul_ps(xmm2, xmm4);
                xmm3 = _mm_mul_ps(xmm3, xmm5);
                xmm0 = _mm_add_ps(xmm0, xmm2);
                xmm1 = _mm_add_ps(xmm1, xmm3);
            }
            k_flt += kernel_w * CH_DT_BLK();
            k_src += src_dh_stride;
        }
    }
        
    if (kernel_flags & KERNEL_FLAG_SUM()) {
        const float *sum_src = PICK_PARAM(const float*, priv_param, SUM_SRC_IDX());
        xmm2 = _mm_loadu_ps(sum_src + 0 * CH_RF_BLK());
        xmm3 = _mm_loadu_ps(sum_src + 1 * CH_RF_BLK());
        xmm0 = _mm_add_ps(xmm0, xmm2);
        xmm1 = _mm_add_ps(xmm1, xmm3);
        PICK_PARAM(const float *, priv_param, SUM_SRC_IDX()) = sum_src + CH_DT_BLK();
    }
    if (kernel_flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
        xmm2 = _mm_setzero_ps();
        xmm0 = _mm_max_ps(xmm0, xmm2);
        xmm1 = _mm_max_ps(xmm1, xmm2);
    }
    if (kernel_flags & KERNEL_FLAG_RELU6()) {
        xmm3 = _mm_set1_ps(6.0f);
        xmm0 = _mm_min_ps(xmm0, xmm3);
        xmm1 = _mm_min_ps(xmm1, xmm3);
    }
    float *dst = PICK_PARAM(float*, priv_param, DST_IDX());
    if (nt_store) {
        _mm_stream_ps(dst + 0 * CH_RF_BLK(), xmm0);
        _mm_stream_ps(dst + 1 * CH_RF_BLK(), xmm1);
    } else {
        _mm_storeu_ps(dst + 0 * CH_RF_BLK(), xmm0);
        _mm_storeu_ps(dst + 1 * CH_RF_BLK(), xmm1);
    }
    PICK_PARAM(const float *, priv_param, SRC_IDX()) += src_sw_stride;
    PICK_PARAM(float *, priv_param, DST_IDX()) = dst + CH_DT_BLK();
}

}}};

#endif
