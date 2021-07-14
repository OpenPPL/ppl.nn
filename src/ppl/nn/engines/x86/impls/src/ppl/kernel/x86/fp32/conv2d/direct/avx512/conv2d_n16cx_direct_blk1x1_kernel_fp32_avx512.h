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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_AVX512_CONV2D_N16CX_DIRECT_BLK1X1_KERNEL_FP32_AVX512_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_AVX512_CONV2D_N16CX_DIRECT_BLK1X1_KERNEL_FP32_AVX512_H_

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/direct/avx512/conv2d_n16cx_direct_kernel_fp32_avx512.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store, int32_t oc_len>
void conv2d_n16cx_direct_fp32_avx512_blk1x1_kernel(
    const int64_t *shar_param,
    int64_t *priv_param)
{
#define IC_COMPUTE_STEP(IC) do {\
    zmm4 = _mm512_set1_ps(ic_src[(IC)]);\
    if (oc_len > 0 * CH_DT_BLK()) zmm0 = _mm512_fmadd_ps(_mm512_loadu_ps(ic_flt + 0 * flt_ocb_stride + (IC) * CH_DT_BLK()), zmm4, zmm0);\
    if (oc_len > 1 * CH_DT_BLK()) zmm1 = _mm512_fmadd_ps(_mm512_loadu_ps(ic_flt + 1 * flt_ocb_stride + (IC) * CH_DT_BLK()), zmm4, zmm1);\
    if (oc_len > 2 * CH_DT_BLK()) zmm2 = _mm512_fmadd_ps(_mm512_loadu_ps(ic_flt + 2 * flt_ocb_stride + (IC) * CH_DT_BLK()), zmm4, zmm2);\
    if (oc_len > 3 * CH_DT_BLK()) zmm3 = _mm512_fmadd_ps(_mm512_loadu_ps(ic_flt + 3 * flt_ocb_stride + (IC) * CH_DT_BLK()), zmm4, zmm3);\
} while (0)

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6;

    const uint64_t kernel_flags = PICK_PARAM(const uint64_t, shar_param, FLAGS_IDX());
    if (kernel_flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
        zmm5 = _mm512_setzero_ps();
    }
    if (kernel_flags & KERNEL_FLAG_RELU6()) {
        zmm6 = _mm512_set1_ps(6.0f);
    }

    if (kernel_flags & KERNEL_FLAG_LD_BIAS()) {
        const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
        if (oc_len > 0 * CH_DT_BLK()) zmm0 = _mm512_loadu_ps(bias + 0 * CH_DT_BLK());
        if (oc_len > 1 * CH_DT_BLK()) zmm1 = _mm512_loadu_ps(bias + 1 * CH_DT_BLK());
        if (oc_len > 2 * CH_DT_BLK()) zmm2 = _mm512_loadu_ps(bias + 2 * CH_DT_BLK());
        if (oc_len > 3 * CH_DT_BLK()) zmm3 = _mm512_loadu_ps(bias + 3 * CH_DT_BLK());
    } else {
        const float* his = PICK_PARAM(const float*, priv_param, HIS_IDX());
        const int64_t his_ocb_stride = shar_param[HIS_OCB_STRIDE_IDX()];
        if (oc_len > 0 * CH_DT_BLK()) zmm0 = _mm512_loadu_ps(his + 0 * his_ocb_stride);
        if (oc_len > 1 * CH_DT_BLK()) zmm1 = _mm512_loadu_ps(his + 1 * his_ocb_stride);
        if (oc_len > 2 * CH_DT_BLK()) zmm2 = _mm512_loadu_ps(his + 2 * his_ocb_stride);
        if (oc_len > 3 * CH_DT_BLK()) zmm3 = _mm512_loadu_ps(his + 3 * his_ocb_stride);
    }

    if (kernel_flags & KERNEL_FLAG_AD_BIAS()) {
        const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
        if (oc_len > 0 * CH_DT_BLK()) zmm0 = _mm512_add_ps(_mm512_loadu_ps(bias + 0 * CH_DT_BLK()), zmm0);
        if (oc_len > 1 * CH_DT_BLK()) zmm1 = _mm512_add_ps(_mm512_loadu_ps(bias + 1 * CH_DT_BLK()), zmm1);
        if (oc_len > 2 * CH_DT_BLK()) zmm2 = _mm512_add_ps(_mm512_loadu_ps(bias + 2 * CH_DT_BLK()), zmm2);
        if (oc_len > 3 * CH_DT_BLK()) zmm3 = _mm512_add_ps(_mm512_loadu_ps(bias + 3 * CH_DT_BLK()), zmm3);
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
        if (oc_len > 0 * CH_DT_BLK()) zmm0 = _mm512_max_ps(zmm0, zmm5);
        if (oc_len > 1 * CH_DT_BLK()) zmm1 = _mm512_max_ps(zmm1, zmm5);
        if (oc_len > 2 * CH_DT_BLK()) zmm2 = _mm512_max_ps(zmm2, zmm5);
        if (oc_len > 3 * CH_DT_BLK()) zmm3 = _mm512_max_ps(zmm3, zmm5);
    }
    if (kernel_flags & KERNEL_FLAG_RELU6()) {
        if (oc_len > 0 * CH_DT_BLK()) zmm0 = _mm512_min_ps(zmm0, zmm6);
        if (oc_len > 1 * CH_DT_BLK()) zmm1 = _mm512_min_ps(zmm1, zmm6);
        if (oc_len > 2 * CH_DT_BLK()) zmm2 = _mm512_min_ps(zmm2, zmm6);
        if (oc_len > 3 * CH_DT_BLK()) zmm3 = _mm512_min_ps(zmm3, zmm6);
    }

    if (nt_store) {
        float* dst = PICK_PARAM(float*, priv_param, DST_IDX());
        const int64_t dst_ocb_stride = shar_param[DST_OCB_STRIDE_IDX()];
        if (oc_len > 0 * CH_DT_BLK()) _mm512_stream_ps(dst + 0 * dst_ocb_stride, zmm0);
        if (oc_len > 1 * CH_DT_BLK()) _mm512_stream_ps(dst + 1 * dst_ocb_stride, zmm1);
        if (oc_len > 2 * CH_DT_BLK()) _mm512_stream_ps(dst + 2 * dst_ocb_stride, zmm2);
        if (oc_len > 3 * CH_DT_BLK()) _mm512_stream_ps(dst + 3 * dst_ocb_stride, zmm3);
    } else {
        float* dst = PICK_PARAM(float*, priv_param, DST_IDX());
        const int64_t dst_ocb_stride = shar_param[DST_OCB_STRIDE_IDX()];
        if (oc_len > 0 * CH_DT_BLK()) _mm512_storeu_ps(dst + 0 * dst_ocb_stride, zmm0);
        if (oc_len > 1 * CH_DT_BLK()) _mm512_storeu_ps(dst + 1 * dst_ocb_stride, zmm1);
        if (oc_len > 2 * CH_DT_BLK()) _mm512_storeu_ps(dst + 2 * dst_ocb_stride, zmm2);
        if (oc_len > 3 * CH_DT_BLK()) _mm512_storeu_ps(dst + 3 * dst_ocb_stride, zmm3);
    }
    PICK_PARAM(const float *, priv_param, SRC_IDX()) += src_sw_stride;
    PICK_PARAM(const float *, priv_param, HIS_IDX()) += CH_DT_BLK();
    PICK_PARAM(float *, priv_param, DST_IDX()) += CH_DT_BLK();
#undef IC_COMPUTE_STEP
}

}}};

#endif
