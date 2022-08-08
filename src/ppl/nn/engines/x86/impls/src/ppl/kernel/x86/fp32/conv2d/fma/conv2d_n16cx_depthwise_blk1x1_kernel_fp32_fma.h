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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DEPTHWISE_FMA_CONV2D_N16CX_DEPTHWISE_BLK1X1_KERNEL_FP32_FMA_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DEPTHWISE_FMA_CONV2D_N16CX_DEPTHWISE_BLK1X1_KERNEL_FP32_FMA_H_

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/fma/conv2d_n16cx_depthwise_kernel_fp32_fma.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store>
void conv2d_n16cx_depthwise_fp32_fma_blk1x1_kernel(
    const int64_t *priv_param,
    const int64_t *shar_param)
{
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5;

    const int64_t src_dh_stride = shar_param[SRC_DH_STRIDE_IDX()];
    const int64_t src_dw_stride = shar_param[SRC_DW_STRIDE_IDX()];
    const int64_t kernel_flags  = shar_param[FLAGS_IDX()];
    const int64_t kernel_w      = shar_param[KW_IDX()];

    const int64_t kh_start = priv_param[KH_START_IDX()];
    const int64_t kh_end   = priv_param[KH_END_IDX()];
    const int64_t kw_start = priv_param[KW_START_IDX()];
    const int64_t kw_end   = priv_param[KW_END_IDX()];

    const float *bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
    ymm0 = _mm256_loadu_ps(bias + 0 * CH_RF_BLK());
    ymm1 = _mm256_loadu_ps(bias + 1 * CH_RF_BLK());

    const float *k_src = PICK_PARAM(const float*, priv_param, SRC_IDX()) + kh_start * src_dh_stride;
    const float *k_flt = PICK_PARAM(const float*, priv_param, FLT_IDX()) + kh_start * kernel_w * CH_DT_BLK();
    if (src_dw_stride == CH_DT_BLK()) {
        for (int32_t kh = kh_start; kh < kh_end; ++kh) {
            for (int32_t kw = kw_start; kw < kw_end; ++kw) {
                ymm2 = _mm256_loadu_ps(k_flt + kw * CH_DT_BLK() + 0 * CH_RF_BLK());
                ymm3 = _mm256_loadu_ps(k_flt + kw * CH_DT_BLK() + 1 * CH_RF_BLK());
                ymm0 = _mm256_fmadd_ps(_mm256_loadu_ps(k_src + kw * CH_DT_BLK() + 0 * CH_RF_BLK()), ymm2, ymm0);
                ymm1 = _mm256_fmadd_ps(_mm256_loadu_ps(k_src + kw * CH_DT_BLK() + 1 * CH_RF_BLK()), ymm3, ymm1);
            }
            k_flt += kernel_w * CH_DT_BLK();
            k_src += src_dh_stride;
        }
    } else {
        for (int32_t kh = kh_start; kh < kh_end; ++kh) {
            for (int32_t kw = kw_start; kw < kw_end; ++kw) {
                ymm2 = _mm256_loadu_ps(k_flt + kw * CH_DT_BLK() + 0 * CH_RF_BLK());
                ymm3 = _mm256_loadu_ps(k_flt + kw * CH_DT_BLK() + 1 * CH_RF_BLK());
                ymm0 = _mm256_fmadd_ps(_mm256_loadu_ps(k_src + kw * src_dw_stride + 0 * CH_RF_BLK()), ymm2, ymm0);
                ymm1 = _mm256_fmadd_ps(_mm256_loadu_ps(k_src + kw * src_dw_stride + 1 * CH_RF_BLK()), ymm3, ymm1);
            }
            k_flt += kernel_w * CH_DT_BLK();
            k_src += src_dh_stride;
        }
    }
        
    if (kernel_flags & KERNEL_FLAG_SUM()) {
        const float *sum_src = PICK_PARAM(const float*, priv_param, SUM_SRC_IDX());
        ymm0 = _mm256_add_ps(_mm256_loadu_ps(sum_src + 0 * CH_RF_BLK()), ymm0);
        ymm1 = _mm256_add_ps(_mm256_loadu_ps(sum_src + 1 * CH_RF_BLK()), ymm1);
    }
    if (kernel_flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
        ymm4 = _mm256_setzero_ps();
        ymm0 = _mm256_max_ps(ymm4, ymm0);
        ymm1 = _mm256_max_ps(ymm4, ymm1);
    }
    if (kernel_flags & KERNEL_FLAG_RELU6()) {
        ymm5 = _mm256_set1_ps(6.0f);
        ymm0 = _mm256_min_ps(ymm5, ymm0);
        ymm1 = _mm256_min_ps(ymm5, ymm1);
    }
    float *dst = PICK_PARAM(float*, priv_param, DST_IDX());
    if (nt_store) {
        _mm256_stream_ps(dst + 0 * CH_RF_BLK(), ymm0);
        _mm256_stream_ps(dst + 1 * CH_RF_BLK(), ymm1);
    } else {
        _mm256_storeu_ps(dst + 0 * CH_RF_BLK(), ymm0);
        _mm256_storeu_ps(dst + 1 * CH_RF_BLK(), ymm1);
    }
}

}}};

#endif
