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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DEPTHWISE_FMA_CONV2D_N16CX_DEPTHWISE_BLK1X7_KERNEL_FP32_FMA_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DEPTHWISE_FMA_CONV2D_N16CX_DEPTHWISE_BLK1X7_KERNEL_FP32_FMA_H_

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/depthwise/fma/conv2d_n16cx_depthwise_kernel_fp32_fma.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store, int32_t spec_stride_w, int32_t w_len>
void conv2d_n16cx_depthwise_fp32_fma_blk1x7_kernel(
    const int64_t *priv_param,
    const int64_t *shar_param)
{
#define KW_COMPUTE_STEP() do {\
    ymm14 = _mm256_loadu_ps(k_flt + 0 * CH_RF_BLK());\
    ymm15 = _mm256_loadu_ps(k_flt + 1 * CH_RF_BLK());\
    if (w_len > 0) {\
        ymm0 = _mm256_fmadd_ps(_mm256_loadu_ps(k_src + 0 * src_sw_stride + 0 * CH_RF_BLK()), ymm14, ymm0);\
        ymm1 = _mm256_fmadd_ps(_mm256_loadu_ps(k_src + 0 * src_sw_stride + 1 * CH_RF_BLK()), ymm15, ymm1);\
    }\
    if (w_len > 1) {\
        ymm2 = _mm256_fmadd_ps(_mm256_loadu_ps(k_src + 1 * src_sw_stride + 0 * CH_RF_BLK()), ymm14, ymm2);\
        ymm3 = _mm256_fmadd_ps(_mm256_loadu_ps(k_src + 1 * src_sw_stride + 1 * CH_RF_BLK()), ymm15, ymm3);\
    }\
    if (w_len > 2) {\
        ymm4 = _mm256_fmadd_ps(_mm256_loadu_ps(k_src + 2 * src_sw_stride + 0 * CH_RF_BLK()), ymm14, ymm4);\
        ymm5 = _mm256_fmadd_ps(_mm256_loadu_ps(k_src + 2 * src_sw_stride + 1 * CH_RF_BLK()), ymm15, ymm5);\
    }\
    if (w_len > 3) {\
        ymm6 = _mm256_fmadd_ps(_mm256_loadu_ps(k_src + 3 * src_sw_stride + 0 * CH_RF_BLK()), ymm14, ymm6);\
        ymm7 = _mm256_fmadd_ps(_mm256_loadu_ps(k_src + 3 * src_sw_stride + 1 * CH_RF_BLK()), ymm15, ymm7);\
    }\
    if (w_len > 4) {\
        ymm8 = _mm256_fmadd_ps(_mm256_loadu_ps(k_src + 4 * src_sw_stride + 0 * CH_RF_BLK()), ymm14, ymm8);\
        ymm9 = _mm256_fmadd_ps(_mm256_loadu_ps(k_src + 4 * src_sw_stride + 1 * CH_RF_BLK()), ymm15, ymm9);\
    }\
    if (w_len > 5) {\
        ymm10 = _mm256_fmadd_ps(_mm256_loadu_ps(k_src + 5 * src_sw_stride + 0 * CH_RF_BLK()), ymm14, ymm10);\
        ymm11 = _mm256_fmadd_ps(_mm256_loadu_ps(k_src + 5 * src_sw_stride + 1 * CH_RF_BLK()), ymm15, ymm11);\
    }\
    if (w_len > 6) {\
        ymm12 = _mm256_fmadd_ps(_mm256_loadu_ps(k_src + 6 * src_sw_stride + 0 * CH_RF_BLK()), ymm14, ymm12);\
        ymm13 = _mm256_fmadd_ps(_mm256_loadu_ps(k_src + 6 * src_sw_stride + 1 * CH_RF_BLK()), ymm15, ymm13);\
    }\
    k_flt += CH_DT_BLK();\
    k_src += src_dw_stride;\
} while (0)

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

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
        if (w_len > 0)  {
            ymm0 = _mm256_loadu_ps(bias + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
            ymm1 = _mm256_loadu_ps(bias + 0 * CH_DT_BLK() + 1 * CH_RF_BLK());
        }
        if (w_len > 1)  {
            ymm2 = ymm0;
            ymm3 = ymm1;
        }
        if (w_len > 2)  {
            ymm4 = ymm0;
            ymm5 = ymm1;
        }
        if (w_len > 3)  {
            ymm6 = ymm0;
            ymm7 = ymm1;
        }
        if (w_len > 4)  {
            ymm8 = ymm0;
            ymm9 = ymm1;
        }
        if (w_len > 5)  {
            ymm10 = ymm0;
            ymm11 = ymm1;
        }
        if (w_len > 6)  {
            ymm12 = ymm0;
            ymm13 = ymm1;
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
                ymm0 = _mm256_add_ps(_mm256_loadu_ps(sum + 0 * CH_DT_BLK() + 0 * CH_RF_BLK()), ymm0);
                ymm1 = _mm256_add_ps(_mm256_loadu_ps(sum + 0 * CH_DT_BLK() + 1 * CH_RF_BLK()), ymm1);
            }
            if (w_len > 1) {
                ymm2 = _mm256_add_ps(_mm256_loadu_ps(sum + 1 * CH_DT_BLK() + 0 * CH_RF_BLK()), ymm2);
                ymm3 = _mm256_add_ps(_mm256_loadu_ps(sum + 1 * CH_DT_BLK() + 1 * CH_RF_BLK()), ymm3);
            }
            if (w_len > 2) {
                ymm4 = _mm256_add_ps(_mm256_loadu_ps(sum + 2 * CH_DT_BLK() + 0 * CH_RF_BLK()), ymm4);
                ymm5 = _mm256_add_ps(_mm256_loadu_ps(sum + 2 * CH_DT_BLK() + 1 * CH_RF_BLK()), ymm5);
            }
            if (w_len > 3) {
                ymm6 = _mm256_add_ps(_mm256_loadu_ps(sum + 3 * CH_DT_BLK() + 0 * CH_RF_BLK()), ymm6);
                ymm7 = _mm256_add_ps(_mm256_loadu_ps(sum + 3 * CH_DT_BLK() + 1 * CH_RF_BLK()), ymm7);
            }
            if (w_len > 4) {
                ymm8 = _mm256_add_ps(_mm256_loadu_ps(sum + 4 * CH_DT_BLK() + 0 * CH_RF_BLK()), ymm8);
                ymm9 = _mm256_add_ps(_mm256_loadu_ps(sum + 4 * CH_DT_BLK() + 1 * CH_RF_BLK()), ymm9);
            }
            if (w_len > 5) {
                ymm10 = _mm256_add_ps(_mm256_loadu_ps(sum + 5 * CH_DT_BLK() + 0 * CH_RF_BLK()), ymm10);
                ymm11 = _mm256_add_ps(_mm256_loadu_ps(sum + 5 * CH_DT_BLK() + 1 * CH_RF_BLK()), ymm11);
            }
            if (w_len > 6) {
                ymm12 = _mm256_add_ps(_mm256_loadu_ps(sum + 6 * CH_DT_BLK() + 0 * CH_RF_BLK()), ymm12);
                ymm13 = _mm256_add_ps(_mm256_loadu_ps(sum + 6 * CH_DT_BLK() + 1 * CH_RF_BLK()), ymm13);
            }
        }
        if (kernel_flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
            ymm14 = _mm256_setzero_ps();
            if (w_len > 0) {
                ymm0 = _mm256_max_ps(ymm0, ymm14);
                ymm1 = _mm256_max_ps(ymm1, ymm14);
            }
            if (w_len > 1) {
                ymm2 = _mm256_max_ps(ymm2, ymm14);
                ymm3 = _mm256_max_ps(ymm3, ymm14);
            }
            if (w_len > 2) {
                ymm4 = _mm256_max_ps(ymm4, ymm14);
                ymm5 = _mm256_max_ps(ymm5, ymm14);
            }
            if (w_len > 3) {
                ymm6 = _mm256_max_ps(ymm6, ymm14);
                ymm7 = _mm256_max_ps(ymm7, ymm14);
            }
            if (w_len > 4) {
                ymm8 = _mm256_max_ps(ymm8, ymm14);
                ymm9 = _mm256_max_ps(ymm9, ymm14);
            }
            if (w_len > 5) {
                ymm10 = _mm256_max_ps(ymm10, ymm14);
                ymm11 = _mm256_max_ps(ymm11, ymm14);
            }
            if (w_len > 6) {
                ymm12 = _mm256_max_ps(ymm12, ymm14);
                ymm13 = _mm256_max_ps(ymm13, ymm14);
            }
        }
        if (kernel_flags & KERNEL_FLAG_RELU6()) {
            ymm15 = _mm256_set1_ps(6.0f);
            if (w_len > 0) {
                ymm0 = _mm256_min_ps(ymm0, ymm15);
                ymm1 = _mm256_min_ps(ymm1, ymm15);
            }
            if (w_len > 1) {
                ymm2 = _mm256_min_ps(ymm2, ymm15);
                ymm3 = _mm256_min_ps(ymm3, ymm15);
            }
            if (w_len > 2) {
                ymm4 = _mm256_min_ps(ymm4, ymm15);
                ymm5 = _mm256_min_ps(ymm5, ymm15);
            }
            if (w_len > 3) {
                ymm6 = _mm256_min_ps(ymm6, ymm15);
                ymm7 = _mm256_min_ps(ymm7, ymm15);
            }
            if (w_len > 4) {
                ymm8 = _mm256_min_ps(ymm8, ymm15);
                ymm9 = _mm256_min_ps(ymm9, ymm15);
            }
            if (w_len > 5) {
                ymm10 = _mm256_min_ps(ymm10, ymm15);
                ymm11 = _mm256_min_ps(ymm11, ymm15);
            }
            if (w_len > 6) {
                ymm12 = _mm256_min_ps(ymm12, ymm15);
                ymm13 = _mm256_min_ps(ymm13, ymm15);
            }
        }
        if (nt_store) {
            if (w_len > 0) {
                _mm256_stream_ps(dst + 0 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm0);
                _mm256_stream_ps(dst + 0 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm1);
            }
            if (w_len > 1) {
                _mm256_stream_ps(dst + 1 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm2);
                _mm256_stream_ps(dst + 1 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm3);
            }
            if (w_len > 2) {
                _mm256_stream_ps(dst + 2 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm4);
                _mm256_stream_ps(dst + 2 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm5);
            }
            if (w_len > 3) {
                _mm256_stream_ps(dst + 3 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm6);
                _mm256_stream_ps(dst + 3 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm7);
            }
            if (w_len > 4) {
                _mm256_stream_ps(dst + 4 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm8);
                _mm256_stream_ps(dst + 4 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm9);
            }
            if (w_len > 5) {
                _mm256_stream_ps(dst + 5 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm10);
                _mm256_stream_ps(dst + 5 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm11);
            }
            if (w_len > 6) {
                _mm256_stream_ps(dst + 6 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm12);
                _mm256_stream_ps(dst + 6 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm13);
            }
        } else {
            if (w_len > 0) {
                _mm256_storeu_ps(dst + 0 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm0);
                _mm256_storeu_ps(dst + 0 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm1);
            }
            if (w_len > 1) {
                _mm256_storeu_ps(dst + 1 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm2);
                _mm256_storeu_ps(dst + 1 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm3);
            }
            if (w_len > 2) {
                _mm256_storeu_ps(dst + 2 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm4);
                _mm256_storeu_ps(dst + 2 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm5);
            }
            if (w_len > 3) {
                _mm256_storeu_ps(dst + 3 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm6);
                _mm256_storeu_ps(dst + 3 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm7);
            }
            if (w_len > 4) {
                _mm256_storeu_ps(dst + 4 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm8);
                _mm256_storeu_ps(dst + 4 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm9);
            }
            if (w_len > 5) {
                _mm256_storeu_ps(dst + 5 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm10);
                _mm256_storeu_ps(dst + 5 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm11);
            }
            if (w_len > 6) {
                _mm256_storeu_ps(dst + 6 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm12);
                _mm256_storeu_ps(dst + 6 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm13);
            }
        }
        src += w_len * src_sw_stride;
        sum += w_len * CH_DT_BLK();
        dst += w_len * CH_DT_BLK();
        ow -= w_len;
    } while (ow > 0);
#undef KW_COMPUTE_STEP
}

}}};

#endif
