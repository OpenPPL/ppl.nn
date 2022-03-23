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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_NDARRAY_AVX512_CONV2D_N16CX_DIRECT_NDARRAY_BLK1X14_KERNEL_FP32_AVX512_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_NDARRAY_AVX512_CONV2D_N16CX_DIRECT_NDARRAY_BLK1X14_KERNEL_FP32_AVX512_H_

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/direct_ndarray/avx512/conv2d_n16cx_direct_ndarray_kernel_fp32_avx512.h"
#include "ppl/kernel/x86/common/array_param_helper.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store, int32_t u_oc, int32_t u_w>
void conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x14_kernel(int64_t *param)
{
#define KW_COMPUTE_STEP() do {\
    if (u_ocb > 0) zmm28 = _mm512_loadu_ps(ic_flt_o16);\
    if (u_ocb > 1) zmm29 = _mm512_loadu_ps(ic_flt_o32);\
    if (u_ocb > 0) _mm_prefetch((const char*)(ic_flt_o16 + OC_DATA_BLK * OC_DATA_BLK), _MM_HINT_T0);\
    if (u_ocb > 1) _mm_prefetch((const char*)(ic_flt_o32 + OC_DATA_BLK * OC_DATA_BLK), _MM_HINT_T0);\
    if (u_w > 0) {\
        zmm30 = _mm512_set1_ps(ic_src_w0[0 * stride_w]);\
        if (u_ocb > 0) zmm0  = _mm512_fmadd_ps(zmm28, zmm30, zmm0);\
        if (u_ocb > 1) zmm14 = _mm512_fmadd_ps(zmm29, zmm30, zmm14);\
    }\
    if (u_w > 1) {\
        zmm31 = _mm512_set1_ps(ic_src_w0[1 * stride_w]);\
        if (u_ocb > 0) zmm1  = _mm512_fmadd_ps(zmm28, zmm31, zmm1);\
        if (u_ocb > 1) zmm15 = _mm512_fmadd_ps(zmm29, zmm31, zmm15);\
    }\
    if (u_w > 2) {\
        zmm30 = _mm512_set1_ps(ic_src_w0[2 * stride_w]);\
        if (u_ocb > 0) zmm2  = _mm512_fmadd_ps(zmm28, zmm30, zmm2);\
        if (u_ocb > 1) zmm16 = _mm512_fmadd_ps(zmm29, zmm30, zmm16);\
    }\
    if (u_w > 3) {\
        zmm31 = _mm512_set1_ps(ic_src_w0[1 * stride_w3]);\
        if (u_ocb > 0) zmm3  = _mm512_fmadd_ps(zmm28, zmm31, zmm3);\
        if (u_ocb > 1) zmm17 = _mm512_fmadd_ps(zmm29, zmm31, zmm17);\
    }\
    if (u_w > 4) {\
        zmm30 = _mm512_set1_ps(ic_src_w0[4 * stride_w]);\
        if (u_ocb > 0) zmm4  = _mm512_fmadd_ps(zmm28, zmm30, zmm4);\
        if (u_ocb > 1) zmm18 = _mm512_fmadd_ps(zmm29, zmm30, zmm18);\
    }\
    if (u_w > 5) {\
        zmm31 = _mm512_set1_ps(ic_src_w5[0 * stride_w]);\
        if (u_ocb > 0) zmm5  = _mm512_fmadd_ps(zmm28, zmm31, zmm5);\
        if (u_ocb > 1) zmm19 = _mm512_fmadd_ps(zmm29, zmm31, zmm19);\
    }\
    if (u_w > 6) {\
        zmm30 = _mm512_set1_ps(ic_src_w5[1 * stride_w]);\
        if (u_ocb > 0) zmm6  = _mm512_fmadd_ps(zmm28, zmm30, zmm6);\
        if (u_ocb > 1) zmm20 = _mm512_fmadd_ps(zmm29, zmm30, zmm20);\
    }\
    if (u_w > 6) {\
        zmm31 = _mm512_set1_ps(ic_src_w5[2 * stride_w]);\
        if (u_ocb > 0) zmm7  = _mm512_fmadd_ps(zmm28, zmm31, zmm7);\
        if (u_ocb > 1) zmm21 = _mm512_fmadd_ps(zmm29, zmm31, zmm21);\
    }\
    if (u_w > 8) {\
        zmm30 = _mm512_set1_ps(ic_src_w5[1 * stride_w3]);\
        if (u_ocb > 0) zmm8  = _mm512_fmadd_ps(zmm28, zmm30, zmm8);\
        if (u_ocb > 1) zmm22 = _mm512_fmadd_ps(zmm29, zmm30, zmm22);\
    }\
    if (u_w > 9) {\
        zmm31 = _mm512_set1_ps(ic_src_w5[4 * stride_w]);\
        if (u_ocb > 0) zmm9  = _mm512_fmadd_ps(zmm28, zmm31, zmm9);\
        if (u_ocb > 1) zmm23 = _mm512_fmadd_ps(zmm29, zmm31, zmm23);\
    }\
    if (u_w > 10) {\
        zmm30 = _mm512_set1_ps(ic_src_w10[0 * stride_w]);\
        if (u_ocb > 0) zmm10 = _mm512_fmadd_ps(zmm28, zmm30, zmm10);\
        if (u_ocb > 1) zmm24 = _mm512_fmadd_ps(zmm29, zmm30, zmm24);\
    }\
    if (u_w > 11) {\
        zmm31 = _mm512_set1_ps(ic_src_w10[1 * stride_w]);\
        if (u_ocb > 0) zmm11 = _mm512_fmadd_ps(zmm28, zmm31, zmm11);\
        if (u_ocb > 1) zmm25 = _mm512_fmadd_ps(zmm29, zmm31, zmm25);\
    }\
    if (u_w > 12) {\
        zmm30 = _mm512_set1_ps(ic_src_w10[2 * stride_w]);\
        if (u_ocb > 0) zmm12 = _mm512_fmadd_ps(zmm28, zmm30, zmm12);\
        if (u_ocb > 1) zmm26 = _mm512_fmadd_ps(zmm29, zmm30, zmm26);\
    }\
    if (u_w > 13) {\
        zmm31 = _mm512_set1_ps(ic_src_w10[1 * stride_w3]);\
        if (u_ocb > 0) zmm13 = _mm512_fmadd_ps(zmm28, zmm31, zmm13);\
        if (u_ocb > 1) zmm27 = _mm512_fmadd_ps(zmm29, zmm31, zmm27);\
    }\
    if (u_ocb > 0) ic_flt_o16 += OC_DATA_BLK;\
    if (u_ocb > 1) ic_flt_o32 += OC_DATA_BLK;\
    if (u_w > 0) ic_src_w0 += 1;\
    if (u_w > 5) ic_src_w5 += 1;\
    if (u_w > 10) ic_src_w10 += 1;\
} while (0)

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;
    __m512 zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;

    const int64_t OC_DATA_BLK = conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::config::OC_DATA_BLK;
    const int64_t u_ocb = div_up(u_oc, OC_DATA_BLK);

    array_param_helper ker_p(param);

    const int64_t kh_start     = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::KH_START_IDX);
    const int64_t kh_end       = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::KH_END_IDX);
    const int64_t kernel_w     = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::KW_IDX);
    const int64_t src_h_stride = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SRC_H_STRIDE_IDX) - kernel_w;
    const int64_t src_c_stride = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SRC_C_STRIDE_IDX) - (kh_end - kh_start) * (src_h_stride + kernel_w);
    const int64_t flt_c_stride = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::FLT_C_STRIDE_IDX) - (kh_end - kh_start) * kernel_w * OC_DATA_BLK;
    const int64_t kernel_flags = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::FLAGS_IDX);

    const int64_t src_offset = kh_start * (src_h_stride + kernel_w);
    const int64_t flt_offset = kh_start * kernel_w * OC_DATA_BLK;

    int64_t dst_w        = ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::DST_WIDTH_IDX);
    const float *src     = ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SRC_PTR_IDX);
    const float *sum_src = ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SUM_SRC_PTR_IDX);
    float *dst           = ker_p.pick<float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::DST_PTR_IDX);
    do {
        const float* bias = ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::BIAS_PTR_IDX);
        if (u_ocb > 0) {
            if (u_w > 0) zmm0 = _mm512_loadu_ps(bias + 0 * OC_DATA_BLK);
            if (u_w > 1) zmm1 = zmm0;
            if (u_w > 2) zmm2 = zmm0;
            if (u_w > 3) zmm3 = zmm0;
            if (u_w > 4) zmm4 = zmm0;
            if (u_w > 5) zmm5 = zmm0;
            if (u_w > 6) zmm6 = zmm0;
            if (u_w > 7) zmm7 = zmm0;
            if (u_w > 8) zmm8 = zmm0;
            if (u_w > 9) zmm9 = zmm0;
            if (u_w > 10) zmm10 = zmm0;
            if (u_w > 11) zmm11 = zmm0;
            if (u_w > 12) zmm12 = zmm0;
            if (u_w > 13) zmm13 = zmm0;
        }
        if (u_ocb > 1) {
            if (u_w > 0) zmm14 = _mm512_loadu_ps(bias + 1 * OC_DATA_BLK);
            if (u_w > 1) zmm15 = zmm14;
            if (u_w > 2) zmm16 = zmm14;
            if (u_w > 3) zmm17 = zmm14;
            if (u_w > 4) zmm18 = zmm14;
            if (u_w > 5) zmm19 = zmm14;
            if (u_w > 6) zmm20 = zmm14;
            if (u_w > 7) zmm21 = zmm14;
            if (u_w > 8) zmm22 = zmm14;
            if (u_w > 9) zmm23 = zmm14;
            if (u_w > 10) zmm24 = zmm14;
            if (u_w > 11) zmm25 = zmm14;
            if (u_w > 12) zmm26 = zmm14;
            if (u_w > 13) zmm27 = zmm14;
        }

        int64_t ic                   = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::CHANNELS_IDX);
        const int64_t stride_w       = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SW_IDX);
        const int64_t flt_ocb_stride = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::FLT_OCB_STRIDE_IDX);
        const int64_t stride_w3      = 3 * stride_w;
        const float *ic_src_w0;
        const float *ic_src_w5;
        const float *ic_src_w10;
        if (u_w > 0) ic_src_w0 = src + src_offset + 0 * stride_w;
        if (u_w > 5) ic_src_w5 = src + src_offset + 5 * stride_w;
        if (u_w > 10) ic_src_w10 = src + src_offset + 10 * stride_w;
        const float *ic_flt_o16;
        const float *ic_flt_o32;
        if (u_ocb > 0) ic_flt_o16 = ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::FLT_PTR_IDX) + flt_offset;
        if (u_ocb > 1) ic_flt_o32 = ic_flt_o16 + flt_ocb_stride;
        while (ic > 0) {
            ic -= 1;
            for (int32_t kh = kh_start; kh < kh_end; ++kh) {
                int32_t kw = kernel_w;
                while (kw > 0) {
                    kw -= 1;
                    KW_COMPUTE_STEP();
                }
                if (u_w > 0) ic_src_w0 += src_h_stride;
                if (u_w > 5) ic_src_w5 += src_h_stride;
                if (u_w > 10) ic_src_w10 += src_h_stride;
            }
            if (u_w > 0) ic_src_w0 += src_c_stride;
            if (u_w > 5) ic_src_w5 += src_c_stride;
            if (u_w > 10) ic_src_w10 += src_c_stride;
            if (u_ocb > 0) ic_flt_o16 += flt_c_stride;
            if (u_ocb > 1) ic_flt_o32 += flt_c_stride;
        }

        if (kernel_flags & conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::flag::SUM) {
            const float *l_sum_src = sum_src;
            const int64_t sum_src_ocb_stride = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SUM_SRC_OCB_STRIDE_IDX);
            if (u_ocb > 0) {
                if (u_w > 0) zmm0 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 0 * OC_DATA_BLK), zmm0);
                if (u_w > 1) zmm1 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 1 * OC_DATA_BLK), zmm1);
                if (u_w > 2) zmm2 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 2 * OC_DATA_BLK), zmm2);
                if (u_w > 3) zmm3 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 3 * OC_DATA_BLK), zmm3);
                if (u_w > 4) zmm4 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 4 * OC_DATA_BLK), zmm4);
                if (u_w > 5) zmm5 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 5 * OC_DATA_BLK), zmm5);
                if (u_w > 6) zmm6 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 6 * OC_DATA_BLK), zmm6);
                if (u_w > 7) zmm7 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 7 * OC_DATA_BLK), zmm7);
                if (u_w > 8) zmm8 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 8 * OC_DATA_BLK), zmm8);
                if (u_w > 9) zmm9 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 9 * OC_DATA_BLK), zmm9);
                if (u_w > 10) zmm10 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 10 * OC_DATA_BLK), zmm10);
                if (u_w > 11) zmm11 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 11 * OC_DATA_BLK), zmm11);
                if (u_w > 12) zmm12 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 12 * OC_DATA_BLK), zmm12);
                if (u_w > 13) zmm13 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 13 * OC_DATA_BLK), zmm13);
            }
            if (u_ocb > 1) {
                l_sum_src += sum_src_ocb_stride;
                if (u_w > 0) zmm14 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 0 * OC_DATA_BLK), zmm14);
                if (u_w > 1) zmm15 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 1 * OC_DATA_BLK), zmm15);
                if (u_w > 2) zmm16 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 2 * OC_DATA_BLK), zmm16);
                if (u_w > 3) zmm17 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 3 * OC_DATA_BLK), zmm17);
                if (u_w > 4) zmm18 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 4 * OC_DATA_BLK), zmm18);
                if (u_w > 5) zmm19 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 5 * OC_DATA_BLK), zmm19);
                if (u_w > 6) zmm20 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 6 * OC_DATA_BLK), zmm20);
                if (u_w > 7) zmm21 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 7 * OC_DATA_BLK), zmm21);
                if (u_w > 8) zmm22 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 8 * OC_DATA_BLK), zmm22);
                if (u_w > 9) zmm23 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 9 * OC_DATA_BLK), zmm23);
                if (u_w > 10) zmm24 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 10 * OC_DATA_BLK), zmm24);
                if (u_w > 11) zmm25 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 11 * OC_DATA_BLK), zmm25);
                if (u_w > 12) zmm26 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 12 * OC_DATA_BLK), zmm26);
                if (u_w > 13) zmm27 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 13 * OC_DATA_BLK), zmm27);
            }
        }
        
        if (kernel_flags & (conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::flag::RELU | conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::flag::RELU6)) {
            zmm30 = _mm512_setzero_ps();
            if (u_ocb > 0) {
                if (u_w > 0) zmm0 = _mm512_max_ps(zmm0, zmm30);
                if (u_w > 1) zmm1 = _mm512_max_ps(zmm1, zmm30);
                if (u_w > 2) zmm2 = _mm512_max_ps(zmm2, zmm30);
                if (u_w > 3) zmm3 = _mm512_max_ps(zmm3, zmm30);
                if (u_w > 4) zmm4 = _mm512_max_ps(zmm4, zmm30);
                if (u_w > 5) zmm5 = _mm512_max_ps(zmm5, zmm30);
                if (u_w > 6) zmm6 = _mm512_max_ps(zmm6, zmm30);
                if (u_w > 7) zmm7 = _mm512_max_ps(zmm7, zmm30);
                if (u_w > 8) zmm8 = _mm512_max_ps(zmm8, zmm30);
                if (u_w > 9) zmm9 = _mm512_max_ps(zmm9, zmm30);
                if (u_w > 10) zmm10 = _mm512_max_ps(zmm10, zmm30);
                if (u_w > 11) zmm11 = _mm512_max_ps(zmm11, zmm30);
                if (u_w > 12) zmm12 = _mm512_max_ps(zmm12, zmm30);
                if (u_w > 13) zmm13 = _mm512_max_ps(zmm13, zmm30);
            }
            if (u_ocb > 1) {
                if (u_w > 0) zmm14 = _mm512_max_ps(zmm14, zmm30);
                if (u_w > 1) zmm15 = _mm512_max_ps(zmm15, zmm30);
                if (u_w > 2) zmm16 = _mm512_max_ps(zmm16, zmm30);
                if (u_w > 3) zmm17 = _mm512_max_ps(zmm17, zmm30);
                if (u_w > 4) zmm18 = _mm512_max_ps(zmm18, zmm30);
                if (u_w > 5) zmm19 = _mm512_max_ps(zmm19, zmm30);
                if (u_w > 6) zmm20 = _mm512_max_ps(zmm20, zmm30);
                if (u_w > 7) zmm21 = _mm512_max_ps(zmm21, zmm30);
                if (u_w > 8) zmm22 = _mm512_max_ps(zmm22, zmm30);
                if (u_w > 9) zmm23 = _mm512_max_ps(zmm23, zmm30);
                if (u_w > 10) zmm24 = _mm512_max_ps(zmm24, zmm30);
                if (u_w > 11) zmm25 = _mm512_max_ps(zmm25, zmm30);
                if (u_w > 12) zmm26 = _mm512_max_ps(zmm26, zmm30);
                if (u_w > 13) zmm27 = _mm512_max_ps(zmm27, zmm30);
            }
        }
        if (kernel_flags & conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::flag::RELU6) {
            zmm31 = _mm512_set1_ps(6.0f);
            if (u_ocb > 0) {
                if (u_w > 0) zmm0 = _mm512_min_ps(zmm0, zmm31);
                if (u_w > 1) zmm1 = _mm512_min_ps(zmm1, zmm31);
                if (u_w > 2) zmm2 = _mm512_min_ps(zmm2, zmm31);
                if (u_w > 3) zmm3 = _mm512_min_ps(zmm3, zmm31);
                if (u_w > 4) zmm4 = _mm512_min_ps(zmm4, zmm31);
                if (u_w > 5) zmm5 = _mm512_min_ps(zmm5, zmm31);
                if (u_w > 6) zmm6 = _mm512_min_ps(zmm6, zmm31);
                if (u_w > 7) zmm7 = _mm512_min_ps(zmm7, zmm31);
                if (u_w > 8) zmm8 = _mm512_min_ps(zmm8, zmm31);
                if (u_w > 9) zmm9 = _mm512_min_ps(zmm9, zmm31);
                if (u_w > 10) zmm10 = _mm512_min_ps(zmm10, zmm31);
                if (u_w > 11) zmm11 = _mm512_min_ps(zmm11, zmm31);
                if (u_w > 12) zmm12 = _mm512_min_ps(zmm12, zmm31);
                if (u_w > 13) zmm13 = _mm512_min_ps(zmm13, zmm31);
            }
            if (u_ocb > 1) {
                if (u_w > 0) zmm14 = _mm512_min_ps(zmm14, zmm31);
                if (u_w > 1) zmm15 = _mm512_min_ps(zmm15, zmm31);
                if (u_w > 2) zmm16 = _mm512_min_ps(zmm16, zmm31);
                if (u_w > 3) zmm17 = _mm512_min_ps(zmm17, zmm31);
                if (u_w > 4) zmm18 = _mm512_min_ps(zmm18, zmm31);
                if (u_w > 5) zmm19 = _mm512_min_ps(zmm19, zmm31);
                if (u_w > 6) zmm20 = _mm512_min_ps(zmm20, zmm31);
                if (u_w > 7) zmm21 = _mm512_min_ps(zmm21, zmm31);
                if (u_w > 8) zmm22 = _mm512_min_ps(zmm22, zmm31);
                if (u_w > 9) zmm23 = _mm512_min_ps(zmm23, zmm31);
                if (u_w > 10) zmm24 = _mm512_min_ps(zmm24, zmm31);
                if (u_w > 11) zmm25 = _mm512_min_ps(zmm25, zmm31);
                if (u_w > 12) zmm26 = _mm512_min_ps(zmm26, zmm31);
                if (u_w > 13) zmm27 = _mm512_min_ps(zmm27, zmm31);
            }
        }

        float* l_dst = dst;
        const int64_t dst_ocb_stride = ker_p.pick<const int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::DST_OCB_STRIDE_IDX);
        if (nt_store) {
            if (u_ocb > 0) {
                if (u_w > 0) _mm512_stream_ps(l_dst + 0 * OC_DATA_BLK, zmm0);
                if (u_w > 1) _mm512_stream_ps(l_dst + 1 * OC_DATA_BLK, zmm1);
                if (u_w > 2) _mm512_stream_ps(l_dst + 2 * OC_DATA_BLK, zmm2);
                if (u_w > 3) _mm512_stream_ps(l_dst + 3 * OC_DATA_BLK, zmm3);
                if (u_w > 4) _mm512_stream_ps(l_dst + 4 * OC_DATA_BLK, zmm4);
                if (u_w > 5) _mm512_stream_ps(l_dst + 5 * OC_DATA_BLK, zmm5);
                if (u_w > 6) _mm512_stream_ps(l_dst + 6 * OC_DATA_BLK, zmm6);
                if (u_w > 7) _mm512_stream_ps(l_dst + 7 * OC_DATA_BLK, zmm7);
                if (u_w > 8) _mm512_stream_ps(l_dst + 8 * OC_DATA_BLK, zmm8);
                if (u_w > 9) _mm512_stream_ps(l_dst + 9 * OC_DATA_BLK, zmm9);
                if (u_w > 10) _mm512_stream_ps(l_dst + 10 * OC_DATA_BLK, zmm10);
                if (u_w > 11) _mm512_stream_ps(l_dst + 11 * OC_DATA_BLK, zmm11);
                if (u_w > 12) _mm512_stream_ps(l_dst + 12 * OC_DATA_BLK, zmm12);
                if (u_w > 13) _mm512_stream_ps(l_dst + 13 * OC_DATA_BLK, zmm13);
            }
            if (u_ocb > 1) {
                l_dst += dst_ocb_stride;
                if (u_w > 0) _mm512_stream_ps(l_dst + 0 * OC_DATA_BLK, zmm14);
                if (u_w > 1) _mm512_stream_ps(l_dst + 1 * OC_DATA_BLK, zmm15);
                if (u_w > 2) _mm512_stream_ps(l_dst + 2 * OC_DATA_BLK, zmm16);
                if (u_w > 3) _mm512_stream_ps(l_dst + 3 * OC_DATA_BLK, zmm17);
                if (u_w > 4) _mm512_stream_ps(l_dst + 4 * OC_DATA_BLK, zmm18);
                if (u_w > 5) _mm512_stream_ps(l_dst + 5 * OC_DATA_BLK, zmm19);
                if (u_w > 6) _mm512_stream_ps(l_dst + 6 * OC_DATA_BLK, zmm20);
                if (u_w > 7) _mm512_stream_ps(l_dst + 7 * OC_DATA_BLK, zmm21);
                if (u_w > 8) _mm512_stream_ps(l_dst + 8 * OC_DATA_BLK, zmm22);
                if (u_w > 9) _mm512_stream_ps(l_dst + 9 * OC_DATA_BLK, zmm23);
                if (u_w > 10) _mm512_stream_ps(l_dst + 10 * OC_DATA_BLK, zmm24);
                if (u_w > 11) _mm512_stream_ps(l_dst + 11 * OC_DATA_BLK, zmm25);
                if (u_w > 12) _mm512_stream_ps(l_dst + 12 * OC_DATA_BLK, zmm26);
                if (u_w > 13) _mm512_stream_ps(l_dst + 13 * OC_DATA_BLK, zmm27);
            }
        } else {
            if (u_ocb > 0) {
                if (u_w > 0) _mm512_storeu_ps(l_dst + 0 * OC_DATA_BLK, zmm0);
                if (u_w > 1) _mm512_storeu_ps(l_dst + 1 * OC_DATA_BLK, zmm1);
                if (u_w > 2) _mm512_storeu_ps(l_dst + 2 * OC_DATA_BLK, zmm2);
                if (u_w > 3) _mm512_storeu_ps(l_dst + 3 * OC_DATA_BLK, zmm3);
                if (u_w > 4) _mm512_storeu_ps(l_dst + 4 * OC_DATA_BLK, zmm4);
                if (u_w > 5) _mm512_storeu_ps(l_dst + 5 * OC_DATA_BLK, zmm5);
                if (u_w > 6) _mm512_storeu_ps(l_dst + 6 * OC_DATA_BLK, zmm6);
                if (u_w > 7) _mm512_storeu_ps(l_dst + 7 * OC_DATA_BLK, zmm7);
                if (u_w > 8) _mm512_storeu_ps(l_dst + 8 * OC_DATA_BLK, zmm8);
                if (u_w > 9) _mm512_storeu_ps(l_dst + 9 * OC_DATA_BLK, zmm9);
                if (u_w > 10) _mm512_storeu_ps(l_dst + 10 * OC_DATA_BLK, zmm10);
                if (u_w > 11) _mm512_storeu_ps(l_dst + 11 * OC_DATA_BLK, zmm11);
                if (u_w > 12) _mm512_storeu_ps(l_dst + 12 * OC_DATA_BLK, zmm12);
                if (u_w > 13) _mm512_storeu_ps(l_dst + 13 * OC_DATA_BLK, zmm13);
            }
            if (u_ocb > 1) {
                l_dst += dst_ocb_stride;
                if (u_w > 0) _mm512_storeu_ps(l_dst + 0 * OC_DATA_BLK, zmm14);
                if (u_w > 1) _mm512_storeu_ps(l_dst + 1 * OC_DATA_BLK, zmm15);
                if (u_w > 2) _mm512_storeu_ps(l_dst + 2 * OC_DATA_BLK, zmm16);
                if (u_w > 3) _mm512_storeu_ps(l_dst + 3 * OC_DATA_BLK, zmm17);
                if (u_w > 4) _mm512_storeu_ps(l_dst + 4 * OC_DATA_BLK, zmm18);
                if (u_w > 5) _mm512_storeu_ps(l_dst + 5 * OC_DATA_BLK, zmm19);
                if (u_w > 6) _mm512_storeu_ps(l_dst + 6 * OC_DATA_BLK, zmm20);
                if (u_w > 7) _mm512_storeu_ps(l_dst + 7 * OC_DATA_BLK, zmm21);
                if (u_w > 8) _mm512_storeu_ps(l_dst + 8 * OC_DATA_BLK, zmm22);
                if (u_w > 9) _mm512_storeu_ps(l_dst + 9 * OC_DATA_BLK, zmm23);
                if (u_w > 10) _mm512_storeu_ps(l_dst + 10 * OC_DATA_BLK, zmm24);
                if (u_w > 11) _mm512_storeu_ps(l_dst + 11 * OC_DATA_BLK, zmm25);
                if (u_w > 12) _mm512_storeu_ps(l_dst + 12 * OC_DATA_BLK, zmm26);
                if (u_w > 13) _mm512_storeu_ps(l_dst + 13 * OC_DATA_BLK, zmm27);
            }
        }
        src += u_w * stride_w;
        sum_src += u_w * OC_DATA_BLK;
        dst += u_w * OC_DATA_BLK;
        dst_w -= u_w;
    } while (dst_w > 0);
    ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SRC_PTR_IDX) = src;
    ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SUM_SRC_PTR_IDX) = sum_src;
    ker_p.pick<float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::DST_PTR_IDX) = dst;
#undef KW_COMPUTE_STEP
}

}}};

#endif
