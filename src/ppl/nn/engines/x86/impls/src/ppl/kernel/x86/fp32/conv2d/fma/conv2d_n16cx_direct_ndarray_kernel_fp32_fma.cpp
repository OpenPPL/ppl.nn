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

#include "ppl/kernel/x86/fp32/conv2d/fma/conv2d_n16cx_direct_ndarray_kernel_fp32_fma.h"
#include "ppl/kernel/x86/common/array_param_helper.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store, int32_t u_oc, int32_t u_w>
void conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel(int64_t *param)
{

#define KW_COMPUTE_STEP() do {\
    if (u_ocr > 0) ymm6 = _mm256_loadu_ps(ic_flt + 0 * OC_REG_ELTS);\
    if (u_ocr > 1) ymm7 = _mm256_loadu_ps(ic_flt + 1 * OC_REG_ELTS);\
    if (u_w > 0) {\
        ymm8 = _mm256_set1_ps(ic_src_w0[0 * stride_w]);\
        if (u_ocr > 0) ymm0 = _mm256_fmadd_ps(ymm6, ymm8, ymm0);\
        if (u_ocr > 1) ymm10 = _mm256_fmadd_ps(ymm7, ymm8, ymm10);\
    }\
    if (u_w > 1) {\
        ymm9 = _mm256_set1_ps(ic_src_w0[1 * stride_w]);\
        if (u_ocr > 0) ymm1 = _mm256_fmadd_ps(ymm6, ymm9, ymm1);\
        if (u_ocr > 1) ymm11 = _mm256_fmadd_ps(ymm7, ymm9, ymm11);\
    }\
    if (u_w > 2) {\
        ymm8 = _mm256_set1_ps(ic_src_w0[2 * stride_w]);\
        if (u_ocr > 0) ymm2 = _mm256_fmadd_ps(ymm6, ymm8, ymm2);\
        if (u_ocr > 1) ymm12 = _mm256_fmadd_ps(ymm7, ymm8, ymm12);\
    }\
    if (u_w > 3) {\
        ymm9 = _mm256_set1_ps(ic_src_w3[0 * stride_w]);\
        if (u_ocr > 0) ymm3 = _mm256_fmadd_ps(ymm6, ymm9, ymm3);\
        if (u_ocr > 1) ymm13 = _mm256_fmadd_ps(ymm7, ymm9, ymm13);\
    }\
    if (u_w > 4) {\
        ymm8 = _mm256_set1_ps(ic_src_w3[1 * stride_w]);\
        if (u_ocr > 0) ymm4 = _mm256_fmadd_ps(ymm6, ymm8, ymm4);\
        if (u_ocr > 1) ymm14 = _mm256_fmadd_ps(ymm7, ymm8, ymm14);\
    }\
    if (u_w > 5) {\
        ymm9 = _mm256_set1_ps(ic_src_w3[2 * stride_w]);\
        if (u_ocr > 0) ymm5 = _mm256_fmadd_ps(ymm6, ymm9, ymm5);\
        if (u_ocr > 1) ymm15 = _mm256_fmadd_ps(ymm7, ymm9, ymm15);\
    }\
    ic_flt += OC_DATA_BLK;\
    if (u_w > 0) ic_src_w0 += 1;\
    if (u_w > 3) ic_src_w3 += 1;\
} while (0)

    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11;
    __m256 ymm12, ymm13, ymm14, ymm15;

    const int64_t OC_DATA_BLK = conv2d_n16cx_direct_ndarray_kernel_fp32_fma::config::OC_DATA_BLK;
    const int64_t OC_REG_ELTS = conv2d_n16cx_direct_ndarray_kernel_fp32_fma::config::OC_REG_ELTS;
    const int64_t u_ocr = div_up(u_oc, OC_REG_ELTS);

    array_param_helper ker_p(param);

    const int64_t kh_start     = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::KH_START_IDX);
    const int64_t kh_end       = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::KH_END_IDX);
    const int64_t kernel_w     = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::KW_IDX);
    const int64_t src_h_stride = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::SRC_H_STRIDE_IDX) - kernel_w;
    const int64_t src_c_stride = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::SRC_C_STRIDE_IDX) - (kh_end - kh_start) * (src_h_stride + kernel_w);
    const int64_t flt_c_stride = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::FLT_C_STRIDE_IDX) - (kh_end - kh_start) * kernel_w * OC_DATA_BLK;
    const int64_t kernel_flags = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::FLAGS_IDX);

    const int64_t src_offset = kh_start * (src_h_stride + kernel_w);
    const int64_t flt_offset = kh_start * kernel_w * OC_DATA_BLK;

    int64_t dst_w        = ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::DST_WIDTH_IDX);
    const float *src     = ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::SRC_PTR_IDX);
    const float *sum_src = ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::SUM_SRC_PTR_IDX);
    float *dst           = ker_p.pick<float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::DST_PTR_IDX);
    do {
        const float *bias = ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::BIAS_PTR_IDX);
        if (u_ocr > 0) {
            if (u_w > 0) ymm0 = _mm256_loadu_ps(bias + 0 * OC_REG_ELTS);
            if (u_w > 1) ymm1 = ymm0;
            if (u_w > 2) ymm2 = ymm0;
            if (u_w > 3) ymm3 = ymm0;
            if (u_w > 4) ymm4 = ymm0;
            if (u_w > 5) ymm5 = ymm0;
        }
        if (u_ocr > 1) {
            if (u_w > 0) ymm10 = _mm256_loadu_ps(bias + 1 * OC_REG_ELTS);
            if (u_w > 1) ymm11 = ymm10;
            if (u_w > 2) ymm12 = ymm10;
            if (u_w > 3) ymm13 = ymm10;
            if (u_w > 4) ymm14 = ymm10;
            if (u_w > 5) ymm15 = ymm10;
        }

        int64_t ic             = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::CHANNELS_IDX);
        const int64_t stride_w = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::SW_IDX);
        const float *ic_src_w0;
        const float *ic_src_w3;
        if (u_w > 0) ic_src_w0 = src + src_offset + 0 * stride_w;
        if (u_w > 3) ic_src_w3 = src + src_offset + 3 * stride_w;
        const float *ic_flt = ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::FLT_PTR_IDX) + flt_offset;
        while (ic > 0) {
            ic -= 1;
            for (int32_t kh = kh_start; kh < kh_end; ++kh) {
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
                if (u_w > 0) ic_src_w0 += src_h_stride;
                if (u_w > 3) ic_src_w3 += src_h_stride;
            }
            if (u_w > 0) ic_src_w0 += src_c_stride;
            if (u_w > 3) ic_src_w3 += src_c_stride;
            ic_flt += flt_c_stride;
        }

        if (kernel_flags & conv2d_n16cx_direct_ndarray_kernel_fp32_fma::flag::SUM) {
            if (u_w > 0) {
                if (u_ocr > 0) ymm0 = _mm256_add_ps(_mm256_loadu_ps(sum_src + 0 * OC_DATA_BLK + 0 * OC_REG_ELTS), ymm0);
                if (u_ocr > 1) ymm10 = _mm256_add_ps(_mm256_loadu_ps(sum_src + 0 * OC_DATA_BLK + 1 * OC_REG_ELTS), ymm10);
            }
            if (u_w > 1) {
                if (u_ocr > 0) ymm1 = _mm256_add_ps(_mm256_loadu_ps(sum_src + 1 * OC_DATA_BLK + 0 * OC_REG_ELTS), ymm1);
                if (u_ocr > 1) ymm11 = _mm256_add_ps(_mm256_loadu_ps(sum_src + 1 * OC_DATA_BLK + 1 * OC_REG_ELTS), ymm11);
            }
            if (u_w > 2) {
                if (u_ocr > 0) ymm2 = _mm256_add_ps(_mm256_loadu_ps(sum_src + 2 * OC_DATA_BLK + 0 * OC_REG_ELTS), ymm2);
                if (u_ocr > 1) ymm12 = _mm256_add_ps(_mm256_loadu_ps(sum_src + 2 * OC_DATA_BLK + 1 * OC_REG_ELTS), ymm12);
            }
            if (u_w > 3) {
                if (u_ocr > 0) ymm3 = _mm256_add_ps(_mm256_loadu_ps(sum_src + 3 * OC_DATA_BLK + 0 * OC_REG_ELTS), ymm3);
                if (u_ocr > 1) ymm13 = _mm256_add_ps(_mm256_loadu_ps(sum_src + 3 * OC_DATA_BLK + 1 * OC_REG_ELTS), ymm13);
            }
            if (u_w > 4) {
                if (u_ocr > 0) ymm4 = _mm256_add_ps(_mm256_loadu_ps(sum_src + 4 * OC_DATA_BLK + 0 * OC_REG_ELTS), ymm4);
                if (u_ocr > 1) ymm14 = _mm256_add_ps(_mm256_loadu_ps(sum_src + 4 * OC_DATA_BLK + 1 * OC_REG_ELTS), ymm14);
            }
            if (u_w > 5) {
                if (u_ocr > 0) ymm5 = _mm256_add_ps(_mm256_loadu_ps(sum_src + 5 * OC_DATA_BLK + 0 * OC_REG_ELTS), ymm5);
                if (u_ocr > 1) ymm15 = _mm256_add_ps(_mm256_loadu_ps(sum_src + 5 * OC_DATA_BLK + 1 * OC_REG_ELTS), ymm15);
            }
        }

        if (kernel_flags & (conv2d_n16cx_direct_ndarray_kernel_fp32_fma::flag::RELU | conv2d_n16cx_direct_ndarray_kernel_fp32_fma::flag::RELU6)) {
            ymm8 = _mm256_setzero_ps();
            if (u_ocr > 0) {
                if (u_w > 0) ymm0 = _mm256_max_ps(ymm0, ymm8);
                if (u_w > 1) ymm1 = _mm256_max_ps(ymm1, ymm8);
                if (u_w > 2) ymm2 = _mm256_max_ps(ymm2, ymm8);
                if (u_w > 3) ymm3 = _mm256_max_ps(ymm3, ymm8);
                if (u_w > 4) ymm4 = _mm256_max_ps(ymm4, ymm8);
                if (u_w > 5) ymm5 = _mm256_max_ps(ymm5, ymm8);
            }
            if (u_ocr > 1) {
                if (u_w > 0) ymm10 = _mm256_max_ps(ymm10, ymm8);
                if (u_w > 1) ymm11 = _mm256_max_ps(ymm11, ymm8);
                if (u_w > 2) ymm12 = _mm256_max_ps(ymm12, ymm8);
                if (u_w > 3) ymm13 = _mm256_max_ps(ymm13, ymm8);
                if (u_w > 4) ymm14 = _mm256_max_ps(ymm14, ymm8);
                if (u_w > 5) ymm15 = _mm256_max_ps(ymm15, ymm8);
            }
        }

        if (kernel_flags & conv2d_n16cx_direct_ndarray_kernel_fp32_fma::flag::RELU6) {
            ymm9 = _mm256_set1_ps(6.0f);
            if (u_ocr > 0) {
                if (u_w > 0) ymm0 = _mm256_min_ps(ymm0, ymm9);
                if (u_w > 1) ymm1 = _mm256_min_ps(ymm1, ymm9);
                if (u_w > 2) ymm2 = _mm256_min_ps(ymm2, ymm9);
                if (u_w > 3) ymm3 = _mm256_min_ps(ymm3, ymm9);
                if (u_w > 4) ymm4 = _mm256_min_ps(ymm4, ymm9);
                if (u_w > 5) ymm5 = _mm256_min_ps(ymm5, ymm9);
            }
            if (u_ocr > 1) {
                if (u_w > 0) ymm10 = _mm256_min_ps(ymm10, ymm9);
                if (u_w > 1) ymm11 = _mm256_min_ps(ymm11, ymm9);
                if (u_w > 2) ymm12 = _mm256_min_ps(ymm12, ymm9);
                if (u_w > 3) ymm13 = _mm256_min_ps(ymm13, ymm9);
                if (u_w > 4) ymm14 = _mm256_min_ps(ymm14, ymm9);
                if (u_w > 5) ymm15 = _mm256_min_ps(ymm15, ymm9);
            }
        }

        if (nt_store) {
            if (u_w > 0) {
                if (u_ocr > 0) _mm256_stream_ps(dst + 0 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm0);
                if (u_ocr > 1) _mm256_stream_ps(dst + 0 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm10);
            }
            if (u_w > 1) {
                if (u_ocr > 0) _mm256_stream_ps(dst + 1 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm1);
                if (u_ocr > 1) _mm256_stream_ps(dst + 1 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm11);
            }
            if (u_w > 2) {
                if (u_ocr > 0) _mm256_stream_ps(dst + 2 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm2);
                if (u_ocr > 1) _mm256_stream_ps(dst + 2 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm12);
            }
            if (u_w > 3) {
                if (u_ocr > 0) _mm256_stream_ps(dst + 3 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm3);
                if (u_ocr > 1) _mm256_stream_ps(dst + 3 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm13);
            }
            if (u_w > 4) {
                if (u_ocr > 0) _mm256_stream_ps(dst + 4 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm4);
                if (u_ocr > 1) _mm256_stream_ps(dst + 4 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm14);
            }
            if (u_w > 5) {
                if (u_ocr > 0) _mm256_stream_ps(dst + 5 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm5);
                if (u_ocr > 1) _mm256_stream_ps(dst + 5 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm15);
            }
        } else {
            if (u_w > 0) {
                if (u_ocr > 0) _mm256_storeu_ps(dst + 0 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm0);
                if (u_ocr > 1) _mm256_storeu_ps(dst + 0 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm10);
            }
            if (u_w > 1) {
                if (u_ocr > 0) _mm256_storeu_ps(dst + 1 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm1);
                if (u_ocr > 1) _mm256_storeu_ps(dst + 1 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm11);
            }
            if (u_w > 2) {
                if (u_ocr > 0) _mm256_storeu_ps(dst + 2 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm2);
                if (u_ocr > 1) _mm256_storeu_ps(dst + 2 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm12);
            }
            if (u_w > 3) {
                if (u_ocr > 0) _mm256_storeu_ps(dst + 3 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm3);
                if (u_ocr > 1) _mm256_storeu_ps(dst + 3 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm13);
            }
            if (u_w > 4) {
                if (u_ocr > 0) _mm256_storeu_ps(dst + 4 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm4);
                if (u_ocr > 1) _mm256_storeu_ps(dst + 4 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm14);
            }
            if (u_w > 5) {
                if (u_ocr > 0) _mm256_storeu_ps(dst + 5 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm5);
                if (u_ocr > 1) _mm256_storeu_ps(dst + 5 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm15);
            }
        }
        { // next block
            dst_w -= u_w;
            src += u_w * stride_w;
            sum_src += u_w * OC_DATA_BLK;
            dst += u_w * OC_DATA_BLK;
        }
    } while (dst_w > 0);
    ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::SRC_PTR_IDX) = src;
    ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::SUM_SRC_PTR_IDX) = sum_src;
    ker_p.pick<float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::DST_PTR_IDX) = dst;
#undef KW_COMPUTE_STEP
}

template <bool nt_store, int32_t u_oc>
inline void conv2d_n16cx_direct_ndarray_fp32_fma_blk1x1_kernel(int64_t *param)
{
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6;

    const int64_t OC_DATA_BLK = conv2d_n16cx_direct_ndarray_kernel_fp32_fma::config::OC_DATA_BLK;
    const int64_t OC_REG_ELTS = conv2d_n16cx_direct_ndarray_kernel_fp32_fma::config::OC_REG_ELTS;
    const int64_t u_ocr = div_up(u_oc, OC_REG_ELTS);

    array_param_helper ker_p(param);

    const int64_t kh_start     = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::KH_START_IDX);
    const int64_t kh_end       = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::KH_END_IDX);
    const int64_t kw_start     = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::KW_START_IDX);
    const int64_t kw_end       = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::KW_END_IDX);
    const int64_t kernel_w     = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::KW_IDX);
    const int64_t stride_w     = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::SW_IDX);
    const int64_t src_h_stride = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::SRC_H_STRIDE_IDX);
    const int64_t src_c_stride = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::SRC_C_STRIDE_IDX) - (kh_end - kh_start) * src_h_stride;
    const int64_t flt_h_stride = kernel_w * OC_DATA_BLK;
    const int64_t flt_c_stride = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::FLT_C_STRIDE_IDX) - (kh_end - kh_start) * flt_h_stride;
    const int64_t kernel_flags = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::FLAGS_IDX);

    const int64_t src_offset = kh_start * src_h_stride;
    const int64_t flt_offset = kh_start * kernel_w * OC_DATA_BLK;

    if (kernel_flags & (conv2d_n16cx_direct_ndarray_kernel_fp32_fma::flag::RELU | conv2d_n16cx_direct_ndarray_kernel_fp32_fma::flag::RELU6)) {
        ymm5 = _mm256_setzero_ps();
    }
    if (kernel_flags & conv2d_n16cx_direct_ndarray_kernel_fp32_fma::flag::RELU6) {
        ymm6 = _mm256_set1_ps(6.0f);
    }

    const float *bias = ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::BIAS_PTR_IDX);
    if (u_ocr > 0) ymm0 = _mm256_loadu_ps(bias + 0 * OC_REG_ELTS);
    if (u_ocr > 1) ymm1 = _mm256_loadu_ps(bias + 1 * OC_REG_ELTS);

    int64_t ic          = ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::CHANNELS_IDX);
    const float *ic_src = ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::SRC_PTR_IDX) + src_offset;
    const float *ic_flt = ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::FLT_PTR_IDX) + flt_offset;
    ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::SRC_PTR_IDX) += stride_w;
    while (ic > 0) {
        ic -= 1;
        for (int32_t kh = kh_start; kh < kh_end; ++kh) {
            for (int32_t kw = kw_start; kw < kw_end; ++kw) {
                if (u_ocr > 0) ymm2 = _mm256_loadu_ps(ic_flt + kw * OC_DATA_BLK + 0 * OC_REG_ELTS);
                if (u_ocr > 1) ymm3 = _mm256_loadu_ps(ic_flt + kw * OC_DATA_BLK + 1 * OC_REG_ELTS);
                ymm4 = _mm256_set1_ps(ic_src[kw]);
                if (u_ocr > 0) ymm0 = _mm256_fmadd_ps(ymm4, ymm2, ymm0);
                if (u_ocr > 1) ymm1 = _mm256_fmadd_ps(ymm4, ymm3, ymm1);
            }
            ic_flt += flt_h_stride;
            ic_src += src_h_stride;
        }
        ic_src += src_c_stride;
        ic_flt += flt_c_stride;
    }

    if (kernel_flags & conv2d_n16cx_direct_ndarray_kernel_fp32_fma::flag::SUM) {
        const float *sum_src = ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::SUM_SRC_PTR_IDX);
        ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::SUM_SRC_PTR_IDX) += OC_DATA_BLK;
        if (u_ocr > 0) ymm0 = _mm256_add_ps(_mm256_loadu_ps(sum_src + 0 * OC_REG_ELTS), ymm0);
        if (u_ocr > 1) ymm1 = _mm256_add_ps(_mm256_loadu_ps(sum_src + 1 * OC_REG_ELTS), ymm1);
    }
    if (kernel_flags & (conv2d_n16cx_direct_ndarray_kernel_fp32_fma::flag::RELU | conv2d_n16cx_direct_ndarray_kernel_fp32_fma::flag::RELU6)) {
        if (u_ocr > 0) ymm0 = _mm256_max_ps(ymm0, ymm5);
        if (u_ocr > 1) ymm1 = _mm256_max_ps(ymm1, ymm5);
    }
    if (kernel_flags & conv2d_n16cx_direct_ndarray_kernel_fp32_fma::flag::RELU6) {
        if (u_ocr > 0) ymm0 = _mm256_min_ps(ymm0, ymm6);
        if (u_ocr > 1) ymm1 = _mm256_min_ps(ymm1, ymm6);
    }

    float *dst = ker_p.pick<float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::DST_PTR_IDX);
    ker_p.pick<float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::DST_PTR_IDX) += OC_DATA_BLK;
    if (nt_store) {
        if (u_ocr > 0) _mm256_stream_ps(dst + 0 * OC_REG_ELTS, ymm0);
        if (u_ocr > 1) _mm256_stream_ps(dst + 1 * OC_REG_ELTS, ymm1);
    } else {
        if (u_ocr > 0) _mm256_storeu_ps(dst + 0 * OC_REG_ELTS, ymm0);
        if (u_ocr > 1) _mm256_storeu_ps(dst + 1 * OC_REG_ELTS, ymm1);
    }
}

#define DIRECT_NDARRAY_BLK1X6_KERNEL_TABLE_BLK(NT_STORE) \
{\
    {\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, 1 * conv2d_n16cx_direct_ndarray_kernel_fp32_fma::config::OC_REG_ELTS, 1>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, 1 * conv2d_n16cx_direct_ndarray_kernel_fp32_fma::config::OC_REG_ELTS, 2>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, 1 * conv2d_n16cx_direct_ndarray_kernel_fp32_fma::config::OC_REG_ELTS, 3>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, 1 * conv2d_n16cx_direct_ndarray_kernel_fp32_fma::config::OC_REG_ELTS, 4>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, 1 * conv2d_n16cx_direct_ndarray_kernel_fp32_fma::config::OC_REG_ELTS, 5>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, 1 * conv2d_n16cx_direct_ndarray_kernel_fp32_fma::config::OC_REG_ELTS, 6>,\
    },\
    {\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, 2 * conv2d_n16cx_direct_ndarray_kernel_fp32_fma::config::OC_REG_ELTS, 1>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, 2 * conv2d_n16cx_direct_ndarray_kernel_fp32_fma::config::OC_REG_ELTS, 2>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, 2 * conv2d_n16cx_direct_ndarray_kernel_fp32_fma::config::OC_REG_ELTS, 3>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, 2 * conv2d_n16cx_direct_ndarray_kernel_fp32_fma::config::OC_REG_ELTS, 4>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, 2 * conv2d_n16cx_direct_ndarray_kernel_fp32_fma::config::OC_REG_ELTS, 5>,\
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x6_kernel<NT_STORE, 2 * conv2d_n16cx_direct_ndarray_kernel_fp32_fma::config::OC_REG_ELTS, 6>,\
    },\
}

const conv2d_n16cx_direct_ndarray_kernel_fp32_fma::func_t
    conv2d_n16cx_direct_ndarray_kernel_fp32_fma::table_[config::NT_STORE_OPT][config::MAX_OC_REGS][config::MAX_W_REGS] =
{
    DIRECT_NDARRAY_BLK1X6_KERNEL_TABLE_BLK(false),
    DIRECT_NDARRAY_BLK1X6_KERNEL_TABLE_BLK(true),
};

const conv2d_n16cx_direct_ndarray_kernel_fp32_fma::func_t
    conv2d_n16cx_direct_ndarray_kernel_fp32_fma::border_table_[config::NT_STORE_OPT][config::MAX_OC_REGS] =
{
    {
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x1_kernel<false, 1 * config::OC_REG_ELTS>,
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x1_kernel<false, 2 * config::OC_REG_ELTS>,
    },
    {
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x1_kernel<true, 1 * config::OC_REG_ELTS>,
        conv2d_n16cx_direct_ndarray_fp32_fma_blk1x1_kernel<true, 2 * config::OC_REG_ELTS>,
    },
};

}}};
