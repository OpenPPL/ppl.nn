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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_FMA_CONV2D_N16CX_DIRECT_BLK1X1_KERNEL_FP32_FMA_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_FMA_CONV2D_N16CX_DIRECT_BLK1X1_KERNEL_FP32_FMA_H_

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/direct/fma/conv2d_n16cx_direct_kernel_fp32_fma.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store, int32_t oc_len>
void conv2d_n16cx_direct_fp32_fma_blk1x1_kernel(
    const int64_t *priv_param,
    const int64_t *shar_param)
{
#define IC_COMPUTE_STEP(IC) do {\
    ymm2 = _mm256_set1_ps(ic_src[(IC)]);\
    if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_fmadd_ps(_mm256_loadu_ps(ic_flt + 0 * CH_RF_BLK() + (IC) * CH_DT_BLK()), ymm2, ymm0);\
    if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_fmadd_ps(_mm256_loadu_ps(ic_flt + 1 * CH_RF_BLK() + (IC) * CH_DT_BLK()), ymm2, ymm1);\
} while (0)

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4;

    const uint64_t kernel_flags = PICK_PARAM(const uint64_t, shar_param, FLAGS_IDX());
    if (kernel_flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
        ymm3 = _mm256_setzero_ps();
    }
    if (kernel_flags & KERNEL_FLAG_RELU6()) {
        ymm4 = _mm256_set1_ps(6.0f);
    }

    if (kernel_flags & KERNEL_FLAG_LD_BIAS()) {
        const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
        if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_loadu_ps(bias + 0 * CH_RF_BLK());
        if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_loadu_ps(bias + 1 * CH_RF_BLK());
    } else {
        const float* his = PICK_PARAM(const float*, priv_param, HIS_IDX());
        if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_loadu_ps(his + 0 * CH_RF_BLK());
        if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_loadu_ps(his + 1 * CH_RF_BLK());
    }

    if (kernel_flags & KERNEL_FLAG_AD_BIAS()) {
        const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
        if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_add_ps(_mm256_loadu_ps(bias + 0 * CH_RF_BLK()), ymm0);
        if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_add_ps(_mm256_loadu_ps(bias + 1 * CH_RF_BLK()), ymm1);
    }

    const int64_t kernel_h = shar_param[KH_IDX()];
    const int64_t kernel_w = shar_param[KW_IDX()];
    const int64_t src_icb_stride = shar_param[SRC_ICB_STRIDE_IDX()];
    const int64_t src_dh_stride = shar_param[SRC_DH_STRIDE_IDX()];
    const int64_t src_dw_stride = shar_param[SRC_DW_STRIDE_IDX()];
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
        if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_max_ps(ymm0, ymm3);
        if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_max_ps(ymm1, ymm3);
    }
    if (kernel_flags & KERNEL_FLAG_RELU6()) {
        if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_min_ps(ymm0, ymm4);
        if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_min_ps(ymm1, ymm4);
    }

    float* dst = PICK_PARAM(float*, priv_param, DST_IDX());
    if (nt_store) {
        if (oc_len > 0 * CH_RF_BLK()) _mm256_stream_ps(dst + 0 * CH_RF_BLK(), ymm0);
        if (oc_len > 1 * CH_RF_BLK()) _mm256_stream_ps(dst + 1 * CH_RF_BLK(), ymm1);
    } else {
        if (oc_len > 0 * CH_RF_BLK()) _mm256_storeu_ps(dst + 0 * CH_RF_BLK(), ymm0);
        if (oc_len > 1 * CH_RF_BLK()) _mm256_storeu_ps(dst + 1 * CH_RF_BLK(), ymm1);
    }
#undef IC_COMPUTE_STEP
}

}}};

#endif
