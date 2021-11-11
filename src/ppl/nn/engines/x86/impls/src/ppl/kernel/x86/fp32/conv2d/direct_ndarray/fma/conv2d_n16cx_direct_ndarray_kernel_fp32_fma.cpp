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

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/direct_ndarray/fma/conv2d_n16cx_direct_ndarray_kernel_fp32_fma.h"

namespace ppl { namespace kernel { namespace x86 {


template <bool nt_store, int32_t spec_kernel_w, int32_t spec_stride_w, int32_t oc_len, int32_t w_len>
void conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel(
    const float *src,
    const float *flt,
    const float *bias,
    const float *sum_src,
    const int64_t ow_len,
    const int64_t kh_start,
    const int64_t kh_end,
    const int64_t src_h_stride,
    const int64_t src_c_stride,
    const int64_t flt_c_stride,
    const conv2d_fp32_param *param,
    float *dst)
{

#define KW_COMPUTE_STEP() do {\
    if (oc_len > 0 * OC_RF_BLK()) ymm6 = _mm256_loadu_ps(k_flt + 0 * OC_RF_BLK());\
    if (oc_len > 1 * OC_RF_BLK()) ymm7 = _mm256_loadu_ps(k_flt + 1 * OC_RF_BLK());\
    if (w_len > 0) {\
        ymm8 = _mm256_set1_ps(k_src[0 * stride_w]);\
        if (oc_len > 0 * OC_RF_BLK()) ymm0 = _mm256_fmadd_ps(ymm6, ymm8, ymm0);\
        if (oc_len > 1 * OC_RF_BLK()) ymm10 = _mm256_fmadd_ps(ymm7, ymm8, ymm10);\
    }\
    if (w_len > 1) {\
        ymm9 = _mm256_set1_ps(k_src[1 * stride_w]);\
        if (oc_len > 0 * OC_RF_BLK()) ymm1 = _mm256_fmadd_ps(ymm6, ymm9, ymm1);\
        if (oc_len > 1 * OC_RF_BLK()) ymm11 = _mm256_fmadd_ps(ymm7, ymm9, ymm11);\
    }\
    if (w_len > 2) {\
        ymm8 = _mm256_set1_ps(k_src[2 * stride_w]);\
        if (oc_len > 0 * OC_RF_BLK()) ymm2 = _mm256_fmadd_ps(ymm6, ymm8, ymm2);\
        if (oc_len > 1 * OC_RF_BLK()) ymm12 = _mm256_fmadd_ps(ymm7, ymm8, ymm12);\
    }\
    if (w_len > 3) {\
        ymm9 = _mm256_set1_ps(k_src[3 * stride_w]);\
        if (oc_len > 0 * OC_RF_BLK()) ymm3 = _mm256_fmadd_ps(ymm6, ymm9, ymm3);\
        if (oc_len > 1 * OC_RF_BLK()) ymm13 = _mm256_fmadd_ps(ymm7, ymm9, ymm13);\
    }\
    if (w_len > 4) {\
        ymm8 = _mm256_set1_ps(k_src[4 * stride_w]);\
        if (oc_len > 0 * OC_RF_BLK()) ymm4 = _mm256_fmadd_ps(ymm6, ymm8, ymm4);\
        if (oc_len > 1 * OC_RF_BLK()) ymm14 = _mm256_fmadd_ps(ymm7, ymm8, ymm14);\
    }\
    if (w_len > 5) {\
        ymm9 = _mm256_set1_ps(k_src[5 * stride_w]);\
        if (oc_len > 0 * OC_RF_BLK()) ymm5 = _mm256_fmadd_ps(ymm6, ymm9, ymm5);\
        if (oc_len > 1 * OC_RF_BLK()) ymm15 = _mm256_fmadd_ps(ymm7, ymm9, ymm15);\
    }\
    k_flt += OC_DT_BLK();\
    k_src += 1;\
} while (0)

    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11;
    __m256 ymm12, ymm13, ymm14, ymm15;

    const int32_t channels = param->channels / param->group;
    const int32_t stride_w = spec_stride_w ? spec_stride_w : param->stride_w;

    int64_t ow             = ow_len;
    const float *w_src     = src;
    const float *w_sum_src = sum_src;
    float *w_dst           = dst;
    do {
        if (oc_len > 0 * OC_RF_BLK()) {
            if (w_len > 0) ymm0 = _mm256_loadu_ps(bias + 0 * OC_RF_BLK());
            if (w_len > 1) ymm1 = ymm0;
            if (w_len > 2) ymm2 = ymm0;
            if (w_len > 3) ymm3 = ymm0;
            if (w_len > 4) ymm4 = ymm0;
            if (w_len > 5) ymm5 = ymm0;
        }
        if (oc_len > 1 * OC_RF_BLK()) {
            if (w_len > 0) ymm10 = _mm256_loadu_ps(bias + 1 * OC_RF_BLK());
            if (w_len > 1) ymm11 = ymm10;
            if (w_len > 2) ymm12 = ymm10;
            if (w_len > 3) ymm13 = ymm10;
            if (w_len > 4) ymm14 = ymm10;
            if (w_len > 5) ymm15 = ymm10;
        }

        const int32_t kernel_w = spec_kernel_w ? spec_kernel_w : param->kernel_w;
        const float *ic_src = w_src + kh_start * src_h_stride;
        const float *ic_flt = flt + kh_start * kernel_w * OC_DT_BLK();
        for (int32_t ic = 0; ic < channels; ++ic) {
            const float *k_src = ic_src;
            const float *k_flt = ic_flt;
            for (int32_t kh = kh_start; kh < kh_end; ++kh) {
                if (spec_kernel_w) {
                    if (spec_kernel_w == 3) {
                        KW_COMPUTE_STEP();
                        KW_COMPUTE_STEP();
                        KW_COMPUTE_STEP();
                    }
                    if (spec_kernel_w == 5) {
                        KW_COMPUTE_STEP();
                        KW_COMPUTE_STEP();
                        KW_COMPUTE_STEP();
                        KW_COMPUTE_STEP();
                    }
                    if (spec_kernel_w == 7) {
                        KW_COMPUTE_STEP();
                        KW_COMPUTE_STEP();
                        KW_COMPUTE_STEP();
                        KW_COMPUTE_STEP();
                        KW_COMPUTE_STEP();
                        KW_COMPUTE_STEP();
                        KW_COMPUTE_STEP();
                    }
                } else {
                    int32_t kw = kernel_w;
                    while (kw >= 3) {
                        kw -= 3;
                        KW_COMPUTE_STEP();
                        KW_COMPUTE_STEP();
                        KW_COMPUTE_STEP();
                    }
                    if (kw & 2) {
                        KW_COMPUTE_STEP();
                        KW_COMPUTE_STEP();
                    }
                    if (kw & 1) {
                        KW_COMPUTE_STEP();
                    }
                }
                k_src += src_h_stride - kernel_w;
            }
            ic_src += src_c_stride;
            ic_flt += flt_c_stride;
        }

        if (param->fuse_flag & conv_fuse_flag::SUM) {
            if (w_len > 0) {
                if (oc_len > 0 * OC_RF_BLK()) ymm0 = _mm256_add_ps(_mm256_loadu_ps(w_sum_src + 0 * OC_DT_BLK() + 0 * OC_RF_BLK()), ymm0);
                if (oc_len > 1 * OC_RF_BLK()) ymm10 = _mm256_add_ps(_mm256_loadu_ps(w_sum_src + 0 * OC_DT_BLK() + 1 * OC_RF_BLK()), ymm10);
            }
            if (w_len > 1) {
                if (oc_len > 0 * OC_RF_BLK()) ymm1 = _mm256_add_ps(_mm256_loadu_ps(w_sum_src + 1 * OC_DT_BLK() + 0 * OC_RF_BLK()), ymm1);
                if (oc_len > 1 * OC_RF_BLK()) ymm11 = _mm256_add_ps(_mm256_loadu_ps(w_sum_src + 1 * OC_DT_BLK() + 1 * OC_RF_BLK()), ymm11);
            }
            if (w_len > 2) {
                if (oc_len > 0 * OC_RF_BLK()) ymm2 = _mm256_add_ps(_mm256_loadu_ps(w_sum_src + 2 * OC_DT_BLK() + 0 * OC_RF_BLK()), ymm2);
                if (oc_len > 1 * OC_RF_BLK()) ymm12 = _mm256_add_ps(_mm256_loadu_ps(w_sum_src + 2 * OC_DT_BLK() + 1 * OC_RF_BLK()), ymm12);
            }
            if (w_len > 3) {
                if (oc_len > 0 * OC_RF_BLK()) ymm3 = _mm256_add_ps(_mm256_loadu_ps(w_sum_src + 3 * OC_DT_BLK() + 0 * OC_RF_BLK()), ymm3);
                if (oc_len > 1 * OC_RF_BLK()) ymm13 = _mm256_add_ps(_mm256_loadu_ps(w_sum_src + 3 * OC_DT_BLK() + 1 * OC_RF_BLK()), ymm13);
            }
            if (w_len > 4) {
                if (oc_len > 0 * OC_RF_BLK()) ymm4 = _mm256_add_ps(_mm256_loadu_ps(w_sum_src + 4 * OC_DT_BLK() + 0 * OC_RF_BLK()), ymm4);
                if (oc_len > 1 * OC_RF_BLK()) ymm14 = _mm256_add_ps(_mm256_loadu_ps(w_sum_src + 4 * OC_DT_BLK() + 1 * OC_RF_BLK()), ymm14);
            }
            if (w_len > 5) {
                if (oc_len > 0 * OC_RF_BLK()) ymm5 = _mm256_add_ps(_mm256_loadu_ps(w_sum_src + 5 * OC_DT_BLK() + 0 * OC_RF_BLK()), ymm5);
                if (oc_len > 1 * OC_RF_BLK()) ymm15 = _mm256_add_ps(_mm256_loadu_ps(w_sum_src + 5 * OC_DT_BLK() + 1 * OC_RF_BLK()), ymm15);
            }
        }

        if (param->fuse_flag & (conv_fuse_flag::RELU | conv_fuse_flag::RELU6)) {
            ymm8 = _mm256_setzero_ps();
            if (oc_len > 0 * OC_RF_BLK()) {
                if (w_len > 0) ymm0 = _mm256_max_ps(ymm0, ymm8);
                if (w_len > 1) ymm1 = _mm256_max_ps(ymm1, ymm8);
                if (w_len > 2) ymm2 = _mm256_max_ps(ymm2, ymm8);
                if (w_len > 3) ymm3 = _mm256_max_ps(ymm3, ymm8);
                if (w_len > 4) ymm4 = _mm256_max_ps(ymm4, ymm8);
                if (w_len > 5) ymm5 = _mm256_max_ps(ymm5, ymm8);
            }
            if (oc_len > 1 * OC_RF_BLK()) {
                if (w_len > 0) ymm10 = _mm256_max_ps(ymm10, ymm8);
                if (w_len > 1) ymm11 = _mm256_max_ps(ymm11, ymm8);
                if (w_len > 2) ymm12 = _mm256_max_ps(ymm12, ymm8);
                if (w_len > 3) ymm13 = _mm256_max_ps(ymm13, ymm8);
                if (w_len > 4) ymm14 = _mm256_max_ps(ymm14, ymm8);
                if (w_len > 5) ymm15 = _mm256_max_ps(ymm15, ymm8);
            }
        }

        if (param->fuse_flag & conv_fuse_flag::RELU6) {
            ymm9 = _mm256_set1_ps(6.0f);
            if (oc_len > 0 * OC_RF_BLK()) {
                if (w_len > 0) ymm0 = _mm256_min_ps(ymm0, ymm9);
                if (w_len > 1) ymm1 = _mm256_min_ps(ymm1, ymm9);
                if (w_len > 2) ymm2 = _mm256_min_ps(ymm2, ymm9);
                if (w_len > 3) ymm3 = _mm256_min_ps(ymm3, ymm9);
                if (w_len > 4) ymm4 = _mm256_min_ps(ymm4, ymm9);
                if (w_len > 5) ymm5 = _mm256_min_ps(ymm5, ymm9);
            }
            if (oc_len > 1 * OC_RF_BLK()) {
                if (w_len > 0) ymm10 = _mm256_min_ps(ymm10, ymm9);
                if (w_len > 1) ymm11 = _mm256_min_ps(ymm11, ymm9);
                if (w_len > 2) ymm12 = _mm256_min_ps(ymm12, ymm9);
                if (w_len > 3) ymm13 = _mm256_min_ps(ymm13, ymm9);
                if (w_len > 4) ymm14 = _mm256_min_ps(ymm14, ymm9);
                if (w_len > 5) ymm15 = _mm256_min_ps(ymm15, ymm9);
            }
        }

        if (nt_store) {
            if (w_len > 0) {
                if (oc_len > 0 * OC_RF_BLK()) _mm256_stream_ps(w_dst + 0 * OC_DT_BLK() + 0 * OC_RF_BLK(), ymm0);
                if (oc_len > 1 * OC_RF_BLK()) _mm256_stream_ps(w_dst + 0 * OC_DT_BLK() + 1 * OC_RF_BLK(), ymm10);
            }
            if (w_len > 1) {
                if (oc_len > 0 * OC_RF_BLK()) _mm256_stream_ps(w_dst + 1 * OC_DT_BLK() + 0 * OC_RF_BLK(), ymm1);
                if (oc_len > 1 * OC_RF_BLK()) _mm256_stream_ps(w_dst + 1 * OC_DT_BLK() + 1 * OC_RF_BLK(), ymm11);
            }
            if (w_len > 2) {
                if (oc_len > 0 * OC_RF_BLK()) _mm256_stream_ps(w_dst + 2 * OC_DT_BLK() + 0 * OC_RF_BLK(), ymm2);
                if (oc_len > 1 * OC_RF_BLK()) _mm256_stream_ps(w_dst + 2 * OC_DT_BLK() + 1 * OC_RF_BLK(), ymm12);
            }
            if (w_len > 3) {
                if (oc_len > 0 * OC_RF_BLK()) _mm256_stream_ps(w_dst + 3 * OC_DT_BLK() + 0 * OC_RF_BLK(), ymm3);
                if (oc_len > 1 * OC_RF_BLK()) _mm256_stream_ps(w_dst + 3 * OC_DT_BLK() + 1 * OC_RF_BLK(), ymm13);
            }
            if (w_len > 4) {
                if (oc_len > 0 * OC_RF_BLK()) _mm256_stream_ps(w_dst + 4 * OC_DT_BLK() + 0 * OC_RF_BLK(), ymm4);
                if (oc_len > 1 * OC_RF_BLK()) _mm256_stream_ps(w_dst + 4 * OC_DT_BLK() + 1 * OC_RF_BLK(), ymm14);
            }
            if (w_len > 5) {
                if (oc_len > 0 * OC_RF_BLK()) _mm256_stream_ps(w_dst + 5 * OC_DT_BLK() + 0 * OC_RF_BLK(), ymm5);
                if (oc_len > 1 * OC_RF_BLK()) _mm256_stream_ps(w_dst + 5 * OC_DT_BLK() + 1 * OC_RF_BLK(), ymm15);
            }
        } else {
            if (w_len > 0) {
                if (oc_len > 0 * OC_RF_BLK()) _mm256_storeu_ps(w_dst + 0 * OC_DT_BLK() + 0 * OC_RF_BLK(), ymm0);
                if (oc_len > 1 * OC_RF_BLK()) _mm256_storeu_ps(w_dst + 0 * OC_DT_BLK() + 1 * OC_RF_BLK(), ymm10);
            }
            if (w_len > 1) {
                if (oc_len > 0 * OC_RF_BLK()) _mm256_storeu_ps(w_dst + 1 * OC_DT_BLK() + 0 * OC_RF_BLK(), ymm1);
                if (oc_len > 1 * OC_RF_BLK()) _mm256_storeu_ps(w_dst + 1 * OC_DT_BLK() + 1 * OC_RF_BLK(), ymm11);
            }
            if (w_len > 2) {
                if (oc_len > 0 * OC_RF_BLK()) _mm256_storeu_ps(w_dst + 2 * OC_DT_BLK() + 0 * OC_RF_BLK(), ymm2);
                if (oc_len > 1 * OC_RF_BLK()) _mm256_storeu_ps(w_dst + 2 * OC_DT_BLK() + 1 * OC_RF_BLK(), ymm12);
            }
            if (w_len > 3) {
                if (oc_len > 0 * OC_RF_BLK()) _mm256_storeu_ps(w_dst + 3 * OC_DT_BLK() + 0 * OC_RF_BLK(), ymm3);
                if (oc_len > 1 * OC_RF_BLK()) _mm256_storeu_ps(w_dst + 3 * OC_DT_BLK() + 1 * OC_RF_BLK(), ymm13);
            }
            if (w_len > 4) {
                if (oc_len > 0 * OC_RF_BLK()) _mm256_storeu_ps(w_dst + 4 * OC_DT_BLK() + 0 * OC_RF_BLK(), ymm4);
                if (oc_len > 1 * OC_RF_BLK()) _mm256_storeu_ps(w_dst + 4 * OC_DT_BLK() + 1 * OC_RF_BLK(), ymm14);
            }
            if (w_len > 5) {
                if (oc_len > 0 * OC_RF_BLK()) _mm256_storeu_ps(w_dst + 5 * OC_DT_BLK() + 0 * OC_RF_BLK(), ymm5);
                if (oc_len > 1 * OC_RF_BLK()) _mm256_storeu_ps(w_dst + 5 * OC_DT_BLK() + 1 * OC_RF_BLK(), ymm15);
            }
        }
        { // next block
            ow -= w_len;
            w_src += w_len * stride_w;
            w_sum_src += w_len * OC_DT_BLK();
            w_dst += w_len * OC_DT_BLK();
        }
    } while (ow > 0);
}

#define DIRECT_NDARRAY_BLK1X6_KERNEL_TABLE_BLK(NT_STORE, KERNEL_W, STRIDE_W) \
{\
    {\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, KERNEL_W, STRIDE_W, 1 * OC_RF_BLK(), 1>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, KERNEL_W, STRIDE_W, 1 * OC_RF_BLK(), 2>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, KERNEL_W, STRIDE_W, 1 * OC_RF_BLK(), 3>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, KERNEL_W, STRIDE_W, 1 * OC_RF_BLK(), 4>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, KERNEL_W, STRIDE_W, 1 * OC_RF_BLK(), 5>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, KERNEL_W, STRIDE_W, 1 * OC_RF_BLK(), 6>,\
    },\
    {\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, KERNEL_W, STRIDE_W, 2 * OC_RF_BLK(), 1>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, KERNEL_W, STRIDE_W, 2 * OC_RF_BLK(), 2>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, KERNEL_W, STRIDE_W, 2 * OC_RF_BLK(), 3>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, KERNEL_W, STRIDE_W, 2 * OC_RF_BLK(), 4>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, KERNEL_W, STRIDE_W, 2 * OC_RF_BLK(), 5>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, KERNEL_W, STRIDE_W, 2 * OC_RF_BLK(), 6>,\
    },\
}

conv2d_n16cx_direct_ndarray_fp32_fma_kernel_func_t
conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel_table[NT_STORE_OPT()][BLK1X6_OC_RF()][BLK1X6_OW_RF()] =
{
    DIRECT_NDARRAY_BLK1X6_KERNEL_TABLE_BLK(false, 0, 0),
    DIRECT_NDARRAY_BLK1X6_KERNEL_TABLE_BLK(true, 0, 0),
};

}}};
