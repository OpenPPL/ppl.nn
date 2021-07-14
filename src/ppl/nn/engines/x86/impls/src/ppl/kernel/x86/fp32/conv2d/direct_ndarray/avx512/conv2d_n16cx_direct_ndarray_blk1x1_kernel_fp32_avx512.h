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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_NDARRAY_AVX512_CONV2D_N16CX_DIRECT_NDARRAY_BLK1X1_KERNEL_FP32_AVX512_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_NDARRAY_AVX512_CONV2D_N16CX_DIRECT_NDARRAY_BLK1X1_KERNEL_FP32_AVX512_H_

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/direct_ndarray/avx512/conv2d_n16cx_direct_ndarray_kernel_fp32_avx512.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store, int32_t oc_len>
void conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x1_kernel(
    const int64_t *shar_param,
    int64_t *priv_param)
{
    __m512 zmm0, zmm1, zmm2, zmm3, zmm4;

    const int64_t kernel_h = shar_param[KH_IDX()];
    const int64_t kernel_w = shar_param[KW_IDX()];
    const int64_t stride_w = shar_param[SW_IDX()];
    const int64_t src_c_stride = shar_param[SRC_C_STRIDE_IDX()];
    const int64_t src_h_stride = shar_param[SRC_H_STRIDE_IDX()];
    const int64_t flt_ocb_stride = shar_param[FLT_OCB_STRIDE_IDX()];
    const int64_t kernel_flags = shar_param[FLAGS_IDX()];
    const int64_t kh_start = priv_param[KH_START_IDX()];
    const int64_t kh_end = priv_param[KH_END_IDX()];
    const int64_t kw_start = priv_param[KW_START_IDX()];
    const int64_t kw_end = priv_param[KW_END_IDX()];

    if (kernel_flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
        zmm3 = _mm512_setzero_ps();
    }
    if (kernel_flags & KERNEL_FLAG_RELU6()) {
        zmm4 = _mm512_set1_ps(6.0f);
    }

    if (kernel_flags & KERNEL_FLAG_LD_BIAS()) {
        const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
        if (oc_len > 0 * OC_DT_BLK()) zmm0 = _mm512_loadu_ps(bias + 0 * OC_DT_BLK());
        if (oc_len > 1 * OC_DT_BLK()) zmm1 = _mm512_loadu_ps(bias + 1 * OC_DT_BLK());
    } else {
        const float* his = PICK_PARAM(const float*, priv_param, HIS_IDX());
        const int64_t his_ocb_stride = shar_param[HIS_OCB_STRIDE_IDX()];
        if (oc_len > 0 * OC_DT_BLK()) zmm0 = _mm512_loadu_ps(his + 0 * his_ocb_stride);
        if (oc_len > 1 * OC_DT_BLK()) zmm1 = _mm512_loadu_ps(his + 1 * his_ocb_stride);
    }

    if (kernel_flags & KERNEL_FLAG_AD_BIAS()) {
        const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
        if (oc_len > 0 * OC_DT_BLK()) zmm0 = _mm512_add_ps(_mm512_loadu_ps(bias + 0 * OC_DT_BLK()), zmm0);
        if (oc_len > 1 * OC_DT_BLK()) zmm1 = _mm512_add_ps(_mm512_loadu_ps(bias + 1 * OC_DT_BLK()), zmm1);
    }

    const float *ic_src = PICK_PARAM(const float*, priv_param, SRC_IDX()) + kh_start * src_h_stride;
    const float *ic_flt = PICK_PARAM(const float*, priv_param, FLT_IDX()) + kh_start * kernel_w * OC_DT_BLK();
    int64_t channels = shar_param[CHANNELS_IDX()];
    do {
        const float *k_src = ic_src;
        const float *k_flt = ic_flt;
        for (int64_t kh = kh_start; kh < kh_end; ++kh) {
            for (int64_t kw = kw_start; kw < kw_end; ++kw) {
                zmm2 = _mm512_set1_ps(k_src[kw]);
                if (oc_len > 0 * OC_DT_BLK()) zmm0 = _mm512_fmadd_ps(_mm512_loadu_ps(k_flt + 0 * flt_ocb_stride + kw * OC_DT_BLK()), zmm2, zmm0);
                if (oc_len > 1 * OC_DT_BLK()) zmm1 = _mm512_fmadd_ps(_mm512_loadu_ps(k_flt + 1 * flt_ocb_stride + kw * OC_DT_BLK()), zmm2, zmm1);
            }
            k_flt += kernel_w * OC_DT_BLK();
            k_src += src_h_stride;
        }
        ic_flt += kernel_h * kernel_w * OC_DT_BLK();
        ic_src += src_c_stride;
        channels -= 1;
    } while (channels > 0);
    
    if (kernel_flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
        if (oc_len > 0 * OC_DT_BLK()) zmm0 = _mm512_max_ps(zmm0, zmm3);
        if (oc_len > 1 * OC_DT_BLK()) zmm1 = _mm512_max_ps(zmm1, zmm3);
    }
    if (kernel_flags & KERNEL_FLAG_RELU6()) {
        if (oc_len > 0 * OC_DT_BLK()) zmm0 = _mm512_min_ps(zmm0, zmm4);
        if (oc_len > 1 * OC_DT_BLK()) zmm1 = _mm512_min_ps(zmm1, zmm4);
    }

    if (nt_store) {
        float* dst = PICK_PARAM(float*, priv_param, DST_IDX());
        const int64_t dst_ocb_stride = shar_param[DST_OCB_STRIDE_IDX()];
        if (oc_len > 0 * OC_DT_BLK()) _mm512_stream_ps(dst + 0 * dst_ocb_stride, zmm0);
        if (oc_len > 1 * OC_DT_BLK()) _mm512_stream_ps(dst + 1 * dst_ocb_stride, zmm1);
    } else {
        float* dst = PICK_PARAM(float*, priv_param, DST_IDX());
        const int64_t dst_ocb_stride = shar_param[DST_OCB_STRIDE_IDX()];
        if (oc_len > 0 * OC_DT_BLK()) _mm512_storeu_ps(dst + 0 * dst_ocb_stride, zmm0);
        if (oc_len > 1 * OC_DT_BLK()) _mm512_storeu_ps(dst + 1 * dst_ocb_stride, zmm1);
    }

    PICK_PARAM(const float*, priv_param, SRC_IDX()) += stride_w;
    PICK_PARAM(const float*, priv_param, HIS_IDX()) += OC_DT_BLK();
    PICK_PARAM(float*, priv_param, DST_IDX()) += OC_DT_BLK();
}

}}};

#endif
