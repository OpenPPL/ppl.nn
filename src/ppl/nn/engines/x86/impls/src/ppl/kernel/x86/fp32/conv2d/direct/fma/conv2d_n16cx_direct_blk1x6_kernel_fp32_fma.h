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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_FMA_CONV2D_N16CX_DIRECT_BLK1X6_KERNEL_FP32_FMA_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_FMA_CONV2D_N16CX_DIRECT_BLK1X6_KERNEL_FP32_FMA_H_

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/direct/fma/conv2d_n16cx_direct_kernel_fp32_fma.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store, int32_t spec_stride_w, int32_t oc_len, int32_t w_len>
void conv2d_n16cx_direct_fp32_fma_blk1x6_kernel(
    const int64_t *priv_param,
    const int64_t *shar_param)
{
#define IC_COMPUTE_STEP(IC) do {\
    if (oc_len > 0 * CH_RF_BLK()) ymm14 = _mm256_loadu_ps(icb_flt + 0 * CH_RF_BLK() + (IC) * CH_DT_BLK());\
    if (oc_len > 1 * CH_RF_BLK()) ymm15 = _mm256_loadu_ps(icb_flt + 1 * CH_RF_BLK() + (IC) * CH_DT_BLK());\
    if (w_len > 0) {\
        ymm12 = _mm256_set1_ps(k_src[(IC) + 0 * src_sw_stride]);\
        if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_fmadd_ps(ymm14, ymm12, ymm0);\
        if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_fmadd_ps(ymm15, ymm12, ymm1);\
    }\
    if (w_len > 1) {\
        ymm13 = _mm256_set1_ps(k_src[(IC) + 1 * src_sw_stride]);\
        if (oc_len > 0 * CH_RF_BLK()) ymm2 = _mm256_fmadd_ps(ymm14, ymm13, ymm2);\
        if (oc_len > 1 * CH_RF_BLK()) ymm3 = _mm256_fmadd_ps(ymm15, ymm13, ymm3);\
    }\
    if (w_len > 2) {\
        ymm12 = _mm256_set1_ps(k_src[(IC) + 2 * src_sw_stride]);\
        if (oc_len > 0 * CH_RF_BLK()) ymm4 = _mm256_fmadd_ps(ymm14, ymm12, ymm4);\
        if (oc_len > 1 * CH_RF_BLK()) ymm5 = _mm256_fmadd_ps(ymm15, ymm12, ymm5);\
    }\
    if (w_len > 3) {\
        ymm13 = _mm256_set1_ps(k_src[(IC) + 3 * src_sw_stride]);\
        if (oc_len > 0 * CH_RF_BLK()) ymm6 = _mm256_fmadd_ps(ymm14, ymm13, ymm6);\
        if (oc_len > 1 * CH_RF_BLK()) ymm7 = _mm256_fmadd_ps(ymm15, ymm13, ymm7);\
    }\
    if (w_len > 4) {\
        ymm12 = _mm256_set1_ps(k_src[(IC) + 4 * src_sw_stride]);\
        if (oc_len > 0 * CH_RF_BLK()) ymm8 = _mm256_fmadd_ps(ymm14, ymm12, ymm8);\
        if (oc_len > 1 * CH_RF_BLK()) ymm9 = _mm256_fmadd_ps(ymm15, ymm12, ymm9);\
    }\
    if (w_len > 5) {\
        ymm13 = _mm256_set1_ps(k_src[(IC) + 5 * src_sw_stride]);\
        if (oc_len > 0 * CH_RF_BLK()) ymm10 = _mm256_fmadd_ps(ymm14, ymm13, ymm10);\
        if (oc_len > 1 * CH_RF_BLK()) ymm11 = _mm256_fmadd_ps(ymm15, ymm13, ymm11);\
    }\
} while (0)

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    const int64_t kernel_h = shar_param[KH_IDX()];
    const int64_t kernel_w = shar_param[KW_IDX()];
    const int64_t src_icb_stride = shar_param[SRC_ICB_STRIDE_IDX()];
    const int64_t flt_icb_stride = (kernel_h - priv_param[KH_END_IDX()] + priv_param[KH_START_IDX()]) * kernel_w * CH_DT_BLK() * CH_DT_BLK();
    const int64_t src_sw_stride = spec_stride_w ? spec_stride_w * CH_DT_BLK() : shar_param[SRC_SW_STRIDE_IDX()];
    const int64_t src_dw_stride = shar_param[SRC_DW_STRIDE_IDX()];
    const int64_t src_dh_stride = shar_param[SRC_DH_STRIDE_IDX()] - kernel_w * src_dw_stride;
    const int64_t kernel_flags = shar_param[FLAGS_IDX()];
    const int64_t kh_start = priv_param[KH_START_IDX()];
    const int64_t kh_end = priv_param[KH_END_IDX()];

    const int64_t src_offset = kh_start * (src_dh_stride + kernel_w * src_dw_stride);
    const int64_t flt_offset = kh_start * kernel_w * CH_DT_BLK() * CH_DT_BLK();

    const float *src = PICK_PARAM(const float*, priv_param, SRC_IDX());
    const float *his = PICK_PARAM(const float*, priv_param, HIS_IDX());
    float *dst       = PICK_PARAM(float*, priv_param, DST_IDX());
    int64_t ow       = priv_param[OW_IDX()];
    do {
        if (kernel_flags & KERNEL_FLAG_LD_BIAS()) {
            const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
            if (w_len > 0) {
                if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_loadu_ps(bias + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_loadu_ps(bias + 1 * CH_RF_BLK());
            }
            if (w_len > 1) {
                if (oc_len > 0 * CH_RF_BLK()) ymm2 = ymm0;
                if (oc_len > 1 * CH_RF_BLK()) ymm3 = ymm1;
            }
            if (w_len > 2) {
                if (oc_len > 0 * CH_RF_BLK()) ymm4 = ymm0;
                if (oc_len > 1 * CH_RF_BLK()) ymm5 = ymm1;
            }
            if (w_len > 3) {
                if (oc_len > 0 * CH_RF_BLK()) ymm6 = ymm0;
                if (oc_len > 1 * CH_RF_BLK()) ymm7 = ymm1;
            }
            if (w_len > 4) {
                if (oc_len > 0 * CH_RF_BLK()) ymm8 = ymm0;
                if (oc_len > 1 * CH_RF_BLK()) ymm9 = ymm1;
            }
            if (w_len > 5) {
                if (oc_len > 0 * CH_RF_BLK()) ymm10 = ymm0;
                if (oc_len > 1 * CH_RF_BLK()) ymm11 = ymm1;
            }
        } else {
            if (w_len > 0) {
                if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_loadu_ps(his + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_loadu_ps(his + 0 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (w_len > 1) {
                if (oc_len > 0 * CH_RF_BLK()) ymm2 = _mm256_loadu_ps(his + 1 * CH_DT_BLK() + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm3 = _mm256_loadu_ps(his + 1 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (w_len > 2) {
                if (oc_len > 0 * CH_RF_BLK()) ymm4 = _mm256_loadu_ps(his + 2 * CH_DT_BLK() + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm5 = _mm256_loadu_ps(his + 2 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (w_len > 3) {
                if (oc_len > 0 * CH_RF_BLK()) ymm6 = _mm256_loadu_ps(his + 3 * CH_DT_BLK() + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm7 = _mm256_loadu_ps(his + 3 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (w_len > 4) {
                if (oc_len > 0 * CH_RF_BLK()) ymm8 = _mm256_loadu_ps(his + 4 * CH_DT_BLK() + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm9 = _mm256_loadu_ps(his + 4 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (w_len > 5) {
                if (oc_len > 0 * CH_RF_BLK()) ymm10 = _mm256_loadu_ps(his + 5 * CH_DT_BLK() + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm11 = _mm256_loadu_ps(his + 5 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
        }

        if (kernel_flags & KERNEL_FLAG_AD_BIAS()) {
            const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
            if (oc_len > 0 * CH_RF_BLK()) ymm14 = _mm256_loadu_ps(bias + 0 * CH_RF_BLK());
            if (oc_len > 1 * CH_RF_BLK()) ymm15 = _mm256_loadu_ps(bias + 1 * CH_RF_BLK());
            if (w_len > 0) {
                if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_add_ps(ymm14, ymm0);
                if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_add_ps(ymm15, ymm1);
            }
            if (w_len > 1) {
                if (oc_len > 0 * CH_RF_BLK()) ymm2 = _mm256_add_ps(ymm14, ymm2);
                if (oc_len > 1 * CH_RF_BLK()) ymm3 = _mm256_add_ps(ymm15, ymm3);
            }
            if (w_len > 2) {
                if (oc_len > 0 * CH_RF_BLK()) ymm4 = _mm256_add_ps(ymm14, ymm4);
                if (oc_len > 1 * CH_RF_BLK()) ymm5 = _mm256_add_ps(ymm15, ymm5);
            }
            if (w_len > 3) {
                if (oc_len > 0 * CH_RF_BLK()) ymm6 = _mm256_add_ps(ymm14, ymm6);
                if (oc_len > 1 * CH_RF_BLK()) ymm7 = _mm256_add_ps(ymm15, ymm7);
            }
            if (w_len > 4) {
                if (oc_len > 0 * CH_RF_BLK()) ymm8 = _mm256_add_ps(ymm14, ymm8);
                if (oc_len > 1 * CH_RF_BLK()) ymm9 = _mm256_add_ps(ymm15, ymm9);
            }
            if (w_len > 5) {
                if (oc_len > 0 * CH_RF_BLK()) ymm10 = _mm256_add_ps(ymm14, ymm10);
                if (oc_len > 1 * CH_RF_BLK()) ymm11 = _mm256_add_ps(ymm15, ymm11);
            }
        }
        
        const float *icb_src = src + src_offset;
        const float *icb_flt = PICK_PARAM(const float*, priv_param, FLT_IDX()) + flt_offset;
        int64_t channels     = shar_param[CHANNELS_IDX()];
        while (channels >= CH_DT_BLK()) {
            channels -= CH_DT_BLK();
            const float *k_src = icb_src;
            for (int64_t kh = kh_start; kh < kh_end; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    for (int64_t ic = 0; ic < CH_DT_BLK(); ic += 4) {
                        IC_COMPUTE_STEP(0);
                        IC_COMPUTE_STEP(1);
                        IC_COMPUTE_STEP(2);
                        IC_COMPUTE_STEP(3);
                        k_src += 4;
                        icb_flt += 4 * CH_DT_BLK();
                    }
                    k_src += src_dw_stride - CH_DT_BLK();
                }
                k_src += src_dh_stride;
            }
            icb_flt += flt_icb_stride;
            icb_src += src_icb_stride;
        }
        if (channels > 0) {
            const float *k_src = icb_src;
            for (int64_t kh = kh_start; kh < kh_end; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    for (int64_t ic = 0; ic < channels; ++ic) {
                        IC_COMPUTE_STEP(0);
                        k_src += 1;
                        icb_flt += CH_DT_BLK();
                    }
                    icb_flt += (CH_DT_BLK() - channels) * CH_DT_BLK();
                    k_src += src_dw_stride - channels;
                }
                k_src += src_dh_stride;
            }
        }
        
        if (kernel_flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
            ymm14 = _mm256_setzero_ps();
            if (w_len > 0) {
                if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_max_ps(ymm0, ymm14);
                if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_max_ps(ymm1, ymm14);
            }
            if (w_len > 1) {
                if (oc_len > 0 * CH_RF_BLK()) ymm2 = _mm256_max_ps(ymm2, ymm14);
                if (oc_len > 1 * CH_RF_BLK()) ymm3 = _mm256_max_ps(ymm3, ymm14);
            }
            if (w_len > 2) {
                if (oc_len > 0 * CH_RF_BLK()) ymm4 = _mm256_max_ps(ymm4, ymm14);
                if (oc_len > 1 * CH_RF_BLK()) ymm5 = _mm256_max_ps(ymm5, ymm14);
            }
            if (w_len > 3) {
                if (oc_len > 0 * CH_RF_BLK()) ymm6 = _mm256_max_ps(ymm6, ymm14);
                if (oc_len > 1 * CH_RF_BLK()) ymm7 = _mm256_max_ps(ymm7, ymm14);
            }
            if (w_len > 4) {
                if (oc_len > 0 * CH_RF_BLK()) ymm8 = _mm256_max_ps(ymm8, ymm14);
                if (oc_len > 1 * CH_RF_BLK()) ymm9 = _mm256_max_ps(ymm9, ymm14);
            }
            if (w_len > 5) {
                if (oc_len > 0 * CH_RF_BLK()) ymm10 = _mm256_max_ps(ymm10, ymm14);
                if (oc_len > 1 * CH_RF_BLK()) ymm11 = _mm256_max_ps(ymm11, ymm14);
            }
        }
        if (kernel_flags & KERNEL_FLAG_RELU6()) {
            ymm15 = _mm256_set1_ps(6.0f);
            if (w_len > 0) {
                if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_min_ps(ymm0, ymm15);
                if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_min_ps(ymm1, ymm15);
            }
            if (w_len > 1) {
                if (oc_len > 0 * CH_RF_BLK()) ymm2 = _mm256_min_ps(ymm2, ymm15);
                if (oc_len > 1 * CH_RF_BLK()) ymm3 = _mm256_min_ps(ymm3, ymm15);
            }
            if (w_len > 2) {
                if (oc_len > 0 * CH_RF_BLK()) ymm4 = _mm256_min_ps(ymm4, ymm15);
                if (oc_len > 1 * CH_RF_BLK()) ymm5 = _mm256_min_ps(ymm5, ymm15);
            }
            if (w_len > 3) {
                if (oc_len > 0 * CH_RF_BLK()) ymm6 = _mm256_min_ps(ymm6, ymm15);
                if (oc_len > 1 * CH_RF_BLK()) ymm7 = _mm256_min_ps(ymm7, ymm15);
            }
            if (w_len > 4) {
                if (oc_len > 0 * CH_RF_BLK()) ymm8 = _mm256_min_ps(ymm8, ymm15);
                if (oc_len > 1 * CH_RF_BLK()) ymm9 = _mm256_min_ps(ymm9, ymm15);
            }
            if (w_len > 5) {
                if (oc_len > 0 * CH_RF_BLK()) ymm10 = _mm256_min_ps(ymm10, ymm15);
                if (oc_len > 1 * CH_RF_BLK()) ymm11 = _mm256_min_ps(ymm11, ymm15);
            }
        }

        if (nt_store) {
            if (w_len > 0) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_stream_ps(dst + 0 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm0);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_stream_ps(dst + 0 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm1);
            }
            if (w_len > 1) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_stream_ps(dst + 1 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm2);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_stream_ps(dst + 1 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm3);
            }
            if (w_len > 2) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_stream_ps(dst + 2 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm4);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_stream_ps(dst + 2 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm5);
            }
            if (w_len > 3) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_stream_ps(dst + 3 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm6);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_stream_ps(dst + 3 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm7);
            }
            if (w_len > 4) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_stream_ps(dst + 4 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm8);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_stream_ps(dst + 4 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm9);
            }
            if (w_len > 5) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_stream_ps(dst + 5 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm10);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_stream_ps(dst + 5 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm11);
            }
        } else {
            if (w_len > 0) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_storeu_ps(dst + 0 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm0);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_storeu_ps(dst + 0 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm1);
            }
            if (w_len > 1) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_storeu_ps(dst + 1 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm2);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_storeu_ps(dst + 1 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm3);
            }
            if (w_len > 2) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_storeu_ps(dst + 2 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm4);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_storeu_ps(dst + 2 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm5);
            }
            if (w_len > 3) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_storeu_ps(dst + 3 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm6);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_storeu_ps(dst + 3 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm7);
            }
            if (w_len > 4) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_storeu_ps(dst + 4 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm8);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_storeu_ps(dst + 4 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm9);
            }
            if (w_len > 5) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_storeu_ps(dst + 5 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm10);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_storeu_ps(dst + 5 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm11);
            }
        }
        src += w_len * src_sw_stride;
        his += w_len * CH_DT_BLK();
        dst += w_len * CH_DT_BLK();
        ow -= w_len;
    } while (ow > 0);
#undef IC_COMPUTE_STEP
}

}}};

#endif
