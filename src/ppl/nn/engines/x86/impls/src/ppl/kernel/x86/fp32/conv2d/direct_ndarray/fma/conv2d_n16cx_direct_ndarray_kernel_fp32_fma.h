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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_NDARRAY_FMA_CONV2D_N16CX_DIRECT_NDARRAY_KERNEL_FP32_FMA_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_NDARRAY_FMA_CONV2D_N16CX_DIRECT_NDARRAY_KERNEL_FP32_FMA_H_

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/conv2d.h"

#define OC_RF_BLK() 8
#define NT_STORE_OPT() 2

#define OC_DT_BLK() 16
#define OC_RF_BLK() 8

#define BLK1X6_OC_RF() 2
#define BLK1X6_OW_RF() 6

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store, int32_t oc_len>
inline void conv2d_n16cx_direct_ndarray_fp32_fma_blk1x1_kernel(
    const float *src,
    const float *flt,
    const float *bias,
    const float *sum_src,
    const int64_t kh_start,
    const int64_t kh_end,
    const int64_t kw_start,
    const int64_t kw_end,
    const int64_t src_h_stride,
    const int64_t src_c_stride,
    const int64_t flt_c_stride,
    const conv2d_fp32_param *param,
    float *dst)
{
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6;

    if (param->fuse_flag & (conv_fuse_flag::RELU | conv_fuse_flag::RELU6)) {
        ymm5 = _mm256_setzero_ps();
    }
    if (param->fuse_flag & conv_fuse_flag::RELU6) {
        ymm6 = _mm256_set1_ps(6.0f);
    }

    if (oc_len > 0 * OC_RF_BLK()) ymm0 = _mm256_loadu_ps(bias + 0 * OC_RF_BLK());
    if (oc_len > 1 * OC_RF_BLK()) ymm1 = _mm256_loadu_ps(bias + 1 * OC_RF_BLK());

    const int32_t kernel_w = param->kernel_w;
    const float *ic_src = src + kh_start * src_h_stride;
    const float *ic_flt = flt + kh_start * kernel_w * OC_DT_BLK();
    for (int32_t ic = 0; ic < param->channels / param->group; ++ic) {
        const float *k_src  = ic_src;
        const float *k_flt  = ic_flt;
        for (int32_t kh = kh_start; kh < kh_end; ++kh) {
            for (int32_t kw = kw_start; kw < kw_end; ++kw) {
                if (oc_len > 0 * OC_RF_BLK()) ymm2 = _mm256_loadu_ps(k_flt + kw * OC_DT_BLK() + 0 * OC_RF_BLK());
                if (oc_len > 1 * OC_RF_BLK()) ymm3 = _mm256_loadu_ps(k_flt + kw * OC_DT_BLK() + 1 * OC_RF_BLK());
                ymm4 = _mm256_set1_ps(k_src[kw]);
                if (oc_len > 0 * OC_RF_BLK()) ymm0 = _mm256_fmadd_ps(ymm4, ymm2, ymm0);
                if (oc_len > 1 * OC_RF_BLK()) ymm1 = _mm256_fmadd_ps(ymm4, ymm3, ymm1);
            }
            k_flt += kernel_w * OC_DT_BLK();
            k_src += src_h_stride;
        }
        ic_src += src_c_stride;
        ic_flt += flt_c_stride;
    }

    if (param->fuse_flag & conv_fuse_flag::SUM) {
        if (oc_len > 0 * OC_RF_BLK()) ymm0 = _mm256_add_ps(_mm256_loadu_ps(sum_src + 0 * OC_RF_BLK()), ymm0);
        if (oc_len > 1 * OC_RF_BLK()) ymm1 = _mm256_add_ps(_mm256_loadu_ps(sum_src + 1 * OC_RF_BLK()), ymm1);
    }
    if (param->fuse_flag & (conv_fuse_flag::RELU | conv_fuse_flag::RELU6)) {
        if (oc_len > 0 * OC_RF_BLK()) ymm0 = _mm256_max_ps(ymm0, ymm5);
        if (oc_len > 1 * OC_RF_BLK()) ymm1 = _mm256_max_ps(ymm1, ymm5);
    }
    if (param->fuse_flag & conv_fuse_flag::RELU6) {
        if (oc_len > 0 * OC_RF_BLK()) ymm0 = _mm256_min_ps(ymm0, ymm6);
        if (oc_len > 1 * OC_RF_BLK()) ymm1 = _mm256_min_ps(ymm1, ymm6);
    }

    if (nt_store) {
        if (oc_len > 0 * OC_RF_BLK()) _mm256_stream_ps(dst + 0 * OC_RF_BLK(), ymm0);
        if (oc_len > 1 * OC_RF_BLK()) _mm256_stream_ps(dst + 1 * OC_RF_BLK(), ymm1);
    } else {
        if (oc_len > 0 * OC_RF_BLK()) _mm256_storeu_ps(dst + 0 * OC_RF_BLK(), ymm0);
        if (oc_len > 1 * OC_RF_BLK()) _mm256_storeu_ps(dst + 1 * OC_RF_BLK(), ymm1);
    }
}

typedef void (*conv2d_n16cx_direct_ndarray_fp32_fma_kernel_func_t)(
    const float *,
    const float *,
    const float *,
    const float *,
    const int64_t,
    const int64_t,
    const int64_t,
    const int64_t,
    const int64_t,
    const int64_t,
    const conv2d_fp32_param *,
    float *);

extern conv2d_n16cx_direct_ndarray_fp32_fma_kernel_func_t
    conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel_table[NT_STORE_OPT()][BLK1X6_OC_RF()][BLK1X6_OW_RF()];

}}}; // namespace ppl::kernel::x86

#endif
