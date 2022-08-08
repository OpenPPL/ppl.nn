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

#include "ppl/kernel/x86/fp32/conv2d/avx512/conv2d_n16cx_direct_ndarray_kernel_fp32_avx512.h"
#include "ppl/kernel/x86/common/array_param_helper.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store, int32_t u_oc>
void conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x1_kernel(int64_t *param)
{
    __m512 zmm0, zmm1, zmm2, zmm3, zmm4;

    const int64_t OC_DATA_BLK = conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::config::OC_DATA_BLK;
    const int64_t u_ocb = div_up(u_oc, OC_DATA_BLK);

    array_param_helper ker_p(param);

    const int64_t kh_start     = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::KH_START_IDX);
    const int64_t kh_end       = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::KH_END_IDX);
    const int64_t kw_start     = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::KW_START_IDX);
    const int64_t kw_end       = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::KW_END_IDX);
    const int64_t kernel_w     = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::KW_IDX);
    const int64_t stride_w     = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SW_IDX);
    const int64_t src_h_stride = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SRC_H_STRIDE_IDX);
    const int64_t src_c_stride = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SRC_C_STRIDE_IDX) - (kh_end - kh_start) * src_h_stride;
    const int64_t flt_h_stride = kernel_w * OC_DATA_BLK;
    const int64_t flt_c_stride = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::FLT_C_STRIDE_IDX) - (kh_end - kh_start) * flt_h_stride;
    const int64_t kernel_flags = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::FLAGS_IDX);

    const int64_t src_offset = kh_start * src_h_stride;
    const int64_t flt_offset = kh_start * kernel_w * OC_DATA_BLK;

    if (kernel_flags & (conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::flag::RELU | conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::flag::RELU6)) {
        zmm3 = _mm512_setzero_ps();
    }
    if (kernel_flags & conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::flag::RELU6) {
        zmm4 = _mm512_set1_ps(6.0f);
    }

    const float *bias = ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::BIAS_PTR_IDX);
    if (u_ocb > 0) zmm0 = _mm512_loadu_ps(bias + 0 * OC_DATA_BLK);
    if (u_ocb > 1) zmm1 = _mm512_loadu_ps(bias + 1 * OC_DATA_BLK);

    int64_t ic                   = ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::CHANNELS_IDX);
    const float *ic_src          = ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SRC_PTR_IDX) + src_offset;
    const float *ic_flt          = ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::FLT_PTR_IDX) + flt_offset;
    const int64_t flt_ocb_stride = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::FLT_OCB_STRIDE_IDX);
    ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SRC_PTR_IDX) += stride_w;
    while (ic > 0) {
        ic -= 1;
        for (int32_t kh = kh_start; kh < kh_end; ++kh) {
            for (int32_t kw = kw_start; kw < kw_end; ++kw) {
                zmm2 = _mm512_set1_ps(ic_src[kw]);
                if (u_ocb > 0) zmm0 = _mm512_fmadd_ps(_mm512_loadu_ps(ic_flt + 0 * flt_ocb_stride + kw * OC_DATA_BLK), zmm2, zmm0);
                if (u_ocb > 1) zmm1 = _mm512_fmadd_ps(_mm512_loadu_ps(ic_flt + 1 * flt_ocb_stride + kw * OC_DATA_BLK), zmm2, zmm1);
            }
            ic_flt += flt_h_stride;
            ic_src += src_h_stride;
        }
        ic_src += src_c_stride;
        ic_flt += flt_c_stride;
    }

    if (kernel_flags & conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::flag::SUM) {
        const float *sum_src = ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SUM_SRC_PTR_IDX);
        const int64_t sum_src_ocb_stride = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SUM_SRC_OCB_STRIDE_IDX);
        ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SUM_SRC_PTR_IDX) += OC_DATA_BLK;
        if (u_ocb > 0) zmm0 = _mm512_add_ps(_mm512_loadu_ps(sum_src + 0 * sum_src_ocb_stride), zmm0);
        if (u_ocb > 1) zmm1 = _mm512_add_ps(_mm512_loadu_ps(sum_src + 1 * sum_src_ocb_stride), zmm1);
    }

    if (kernel_flags & (conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::flag::RELU | conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::flag::RELU6)) {
        if (u_ocb > 0) zmm0 = _mm512_max_ps(zmm0, zmm3);
        if (u_ocb > 1) zmm1 = _mm512_max_ps(zmm1, zmm3);
    }
    if (kernel_flags & conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::flag::RELU6) {
        if (u_ocb > 0) zmm0 = _mm512_min_ps(zmm0, zmm4);
        if (u_ocb > 1) zmm1 = _mm512_min_ps(zmm1, zmm4);
    }

    float *dst = ker_p.pick<float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::DST_PTR_IDX);
    const int64_t dst_ocb_stride = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::DST_OCB_STRIDE_IDX);
    ker_p.pick<float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::DST_PTR_IDX) += OC_DATA_BLK;
    if (nt_store) {
        if (u_ocb > 0) _mm512_stream_ps(dst + 0 * dst_ocb_stride, zmm0);
        if (u_ocb > 1) _mm512_stream_ps(dst + 1 * dst_ocb_stride, zmm1);
    } else {
        if (u_ocb > 0) _mm512_storeu_ps(dst + 0 * dst_ocb_stride, zmm0);
        if (u_ocb > 1) _mm512_storeu_ps(dst + 1 * dst_ocb_stride, zmm1);
    }
}

}}};

#endif
