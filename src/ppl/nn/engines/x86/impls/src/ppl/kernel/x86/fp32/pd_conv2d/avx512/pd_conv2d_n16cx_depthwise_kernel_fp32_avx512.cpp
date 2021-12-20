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

#include "ppl/kernel/x86/fp32/pd_conv2d/avx512/pd_conv2d_n16cx_depthwise_kernel_fp32_avx512.h"
#include "ppl/kernel/x86/common/array_param_helper.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store, int32_t spec_stride_w, int32_t u_w>
void pd_conv2d_n16cx_depthwise_fp32_avx512_blk1x31_kernel(int64_t *param)
{
#define KW_COMPUTE_STEP() do {\
    zmm31 = _mm512_loadu_ps(flt);\
    if (u_w > 0) zmm0 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 0 * src_sw_stride), zmm31, zmm0);\
    if (u_w > 1) zmm1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 1 * src_sw_stride), zmm31, zmm1);\
    if (u_w > 2) zmm2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 2 * src_sw_stride), zmm31, zmm2);\
    if (u_w > 3) zmm3 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 3 * src_sw_stride), zmm31, zmm3);\
    if (u_w > 4) zmm4 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 4 * src_sw_stride), zmm31, zmm4);\
    if (u_w > 5) zmm5 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 5 * src_sw_stride), zmm31, zmm5);\
    if (u_w > 6) zmm6 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 6 * src_sw_stride), zmm31, zmm6);\
    if (u_w > 7) zmm7 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 7 * src_sw_stride), zmm31, zmm7);\
    if (u_w > 8) zmm8 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 8 * src_sw_stride), zmm31, zmm8);\
    if (u_w > 9) zmm9 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 9 * src_sw_stride), zmm31, zmm9);\
    if (u_w > 10) zmm10 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 10 * src_sw_stride), zmm31, zmm10);\
    if (u_w > 11) zmm11 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 11 * src_sw_stride), zmm31, zmm11);\
    if (u_w > 12) zmm12 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 12 * src_sw_stride), zmm31, zmm12);\
    if (u_w > 13) zmm13 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 13 * src_sw_stride), zmm31, zmm13);\
    if (u_w > 14) zmm14 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 14 * src_sw_stride), zmm31, zmm14);\
    if (u_w > 15) zmm15 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 15 * src_sw_stride), zmm31, zmm15);\
    if (u_w > 16) zmm16 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 16 * src_sw_stride), zmm31, zmm16);\
    if (u_w > 17) zmm17 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 17 * src_sw_stride), zmm31, zmm17);\
    if (u_w > 18) zmm18 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 18 * src_sw_stride), zmm31, zmm18);\
    if (u_w > 19) zmm19 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 19 * src_sw_stride), zmm31, zmm19);\
    if (u_w > 20) zmm20 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 20 * src_sw_stride), zmm31, zmm20);\
    if (u_w > 21) zmm21 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 21 * src_sw_stride), zmm31, zmm21);\
    if (u_w > 22) zmm22 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 22 * src_sw_stride), zmm31, zmm22);\
    if (u_w > 23) zmm23 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 23 * src_sw_stride), zmm31, zmm23);\
    if (u_w > 24) zmm24 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 24 * src_sw_stride), zmm31, zmm24);\
    if (u_w > 25) zmm25 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 25 * src_sw_stride), zmm31, zmm25);\
    if (u_w > 26) zmm26 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 26 * src_sw_stride), zmm31, zmm26);\
    if (u_w > 27) zmm27 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 27 * src_sw_stride), zmm31, zmm27);\
    if (u_w > 28) zmm28 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 28 * src_sw_stride), zmm31, zmm28);\
    if (u_w > 29) zmm29 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 29 * src_sw_stride), zmm31, zmm29);\
    if (u_w > 30) zmm30 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 30 * src_sw_stride), zmm31, zmm30);\
    flt += CH_DATA_BLK;\
    src += CH_DATA_BLK;\
} while (0)

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;
    __m512 zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;

    const int64_t CH_DATA_BLK = pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::config::CH_DATA_BLK;

    array_param_helper ker_p(param);

    const int64_t src_sw_stride = spec_stride_w ? spec_stride_w * CH_DATA_BLK : 
                                  ker_p.pick<const int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::SRC_SW_STRIDE_IDX);
    const int64_t kernel_flags  = ker_p.pick<const int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::FLAGS_IDX);
    const int64_t kernel_w      = ker_p.pick<const int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::KW_IDX);

    const int64_t kh_start = ker_p.pick<const int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::KH_START_IDX);
    const int64_t kh_end   = ker_p.pick<const int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::KH_END_IDX);

    const int64_t src_uw_stride = u_w * src_sw_stride;
    const int64_t flt_offset    = kh_start * kernel_w * CH_DATA_BLK;

    const float **src_kh_list = ker_p.pick<const float**>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::SRC_PTR_KH_LIST_IDX);
    float *dst                = ker_p.pick<float*>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::DST_PTR_IDX);
    int64_t dst_w             = ker_p.pick<const int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::DST_WIDTH_IDX);
    do {
        const float* bias = ker_p.pick<const float*>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::BIAS_PTR_IDX);
        if (u_w > 0) zmm0 = _mm512_loadu_ps(bias + 0 * CH_DATA_BLK);
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
        if (u_w > 14) zmm14 = zmm0;
        if (u_w > 15) zmm15 = zmm0;
        if (u_w > 16) zmm16 = zmm0;
        if (u_w > 17) zmm17 = zmm0;
        if (u_w > 18) zmm18 = zmm0;
        if (u_w > 19) zmm19 = zmm0;
        if (u_w > 20) zmm20 = zmm0;
        if (u_w > 21) zmm21 = zmm0;
        if (u_w > 22) zmm22 = zmm0;
        if (u_w > 23) zmm23 = zmm0;
        if (u_w > 24) zmm24 = zmm0;
        if (u_w > 25) zmm25 = zmm0;
        if (u_w > 26) zmm26 = zmm0;
        if (u_w > 27) zmm27 = zmm0;
        if (u_w > 28) zmm28 = zmm0;
        if (u_w > 29) zmm29 = zmm0;
        if (u_w > 30) zmm30 = zmm0;

        const float *flt = ker_p.pick<const float*>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::FLT_PTR_IDX) + flt_offset;
        for (int32_t kh = kh_start; kh < kh_end; ++kh) {
            const float *src = src_kh_list[kh];
            src_kh_list[kh] = src + src_uw_stride;
            for (int32_t kw = 0; kw < kernel_w; ++kw) {
                KW_COMPUTE_STEP();
            }
        }

        if (kernel_flags & (pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::flag::RELU | pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::flag::RELU6)) {
            zmm31 = _mm512_setzero_ps();
            if (u_w > 0) zmm0 = _mm512_max_ps(zmm0, zmm31);
            if (u_w > 1) zmm1 = _mm512_max_ps(zmm1, zmm31);
            if (u_w > 2) zmm2 = _mm512_max_ps(zmm2, zmm31);
            if (u_w > 3) zmm3 = _mm512_max_ps(zmm3, zmm31);
            if (u_w > 4) zmm4 = _mm512_max_ps(zmm4, zmm31);
            if (u_w > 5) zmm5 = _mm512_max_ps(zmm5, zmm31);
            if (u_w > 6) zmm6 = _mm512_max_ps(zmm6, zmm31);
            if (u_w > 7) zmm7 = _mm512_max_ps(zmm7, zmm31);
            if (u_w > 8) zmm8 = _mm512_max_ps(zmm8, zmm31);
            if (u_w > 9) zmm9 = _mm512_max_ps(zmm9, zmm31);
            if (u_w > 10) zmm10 = _mm512_max_ps(zmm10, zmm31);
            if (u_w > 11) zmm11 = _mm512_max_ps(zmm11, zmm31);
            if (u_w > 12) zmm12 = _mm512_max_ps(zmm12, zmm31);
            if (u_w > 13) zmm13 = _mm512_max_ps(zmm13, zmm31);
            if (u_w > 14) zmm14 = _mm512_max_ps(zmm14, zmm31);
            if (u_w > 15) zmm15 = _mm512_max_ps(zmm15, zmm31);
            if (u_w > 16) zmm16 = _mm512_max_ps(zmm16, zmm31);
            if (u_w > 17) zmm17 = _mm512_max_ps(zmm17, zmm31);
            if (u_w > 18) zmm18 = _mm512_max_ps(zmm18, zmm31);
            if (u_w > 19) zmm19 = _mm512_max_ps(zmm19, zmm31);
            if (u_w > 20) zmm20 = _mm512_max_ps(zmm20, zmm31);
            if (u_w > 21) zmm21 = _mm512_max_ps(zmm21, zmm31);
            if (u_w > 22) zmm22 = _mm512_max_ps(zmm22, zmm31);
            if (u_w > 23) zmm23 = _mm512_max_ps(zmm23, zmm31);
            if (u_w > 24) zmm24 = _mm512_max_ps(zmm24, zmm31);
            if (u_w > 25) zmm25 = _mm512_max_ps(zmm25, zmm31);
            if (u_w > 26) zmm26 = _mm512_max_ps(zmm26, zmm31);
            if (u_w > 27) zmm27 = _mm512_max_ps(zmm27, zmm31);
            if (u_w > 28) zmm28 = _mm512_max_ps(zmm28, zmm31);
            if (u_w > 29) zmm29 = _mm512_max_ps(zmm29, zmm31);
            if (u_w > 30) zmm30 = _mm512_max_ps(zmm30, zmm31);
        }
        if (kernel_flags & pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::flag::RELU6) {
            zmm31 = _mm512_set1_ps(6.0f);
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
            if (u_w > 14) zmm14 = _mm512_min_ps(zmm14, zmm31);
            if (u_w > 15) zmm15 = _mm512_min_ps(zmm15, zmm31);
            if (u_w > 16) zmm16 = _mm512_min_ps(zmm16, zmm31);
            if (u_w > 17) zmm17 = _mm512_min_ps(zmm17, zmm31);
            if (u_w > 18) zmm18 = _mm512_min_ps(zmm18, zmm31);
            if (u_w > 19) zmm19 = _mm512_min_ps(zmm19, zmm31);
            if (u_w > 20) zmm20 = _mm512_min_ps(zmm20, zmm31);
            if (u_w > 21) zmm21 = _mm512_min_ps(zmm21, zmm31);
            if (u_w > 22) zmm22 = _mm512_min_ps(zmm22, zmm31);
            if (u_w > 23) zmm23 = _mm512_min_ps(zmm23, zmm31);
            if (u_w > 24) zmm24 = _mm512_min_ps(zmm24, zmm31);
            if (u_w > 25) zmm25 = _mm512_min_ps(zmm25, zmm31);
            if (u_w > 26) zmm26 = _mm512_min_ps(zmm26, zmm31);
            if (u_w > 27) zmm27 = _mm512_min_ps(zmm27, zmm31);
            if (u_w > 28) zmm28 = _mm512_min_ps(zmm28, zmm31);
            if (u_w > 29) zmm29 = _mm512_min_ps(zmm29, zmm31);
            if (u_w > 30) zmm30 = _mm512_min_ps(zmm30, zmm31);
        }
        if (nt_store) {
            if (u_w > 0) _mm512_stream_ps(dst + 0 * CH_DATA_BLK, zmm0);
            if (u_w > 1) _mm512_stream_ps(dst + 1 * CH_DATA_BLK, zmm1);
            if (u_w > 2) _mm512_stream_ps(dst + 2 * CH_DATA_BLK, zmm2);
            if (u_w > 3) _mm512_stream_ps(dst + 3 * CH_DATA_BLK, zmm3);
            if (u_w > 4) _mm512_stream_ps(dst + 4 * CH_DATA_BLK, zmm4);
            if (u_w > 5) _mm512_stream_ps(dst + 5 * CH_DATA_BLK, zmm5);
            if (u_w > 6) _mm512_stream_ps(dst + 6 * CH_DATA_BLK, zmm6);
            if (u_w > 7) _mm512_stream_ps(dst + 7 * CH_DATA_BLK, zmm7);
            if (u_w > 8) _mm512_stream_ps(dst + 8 * CH_DATA_BLK, zmm8);
            if (u_w > 9) _mm512_stream_ps(dst + 9 * CH_DATA_BLK, zmm9);
            if (u_w > 10) _mm512_stream_ps(dst + 10 * CH_DATA_BLK, zmm10);
            if (u_w > 11) _mm512_stream_ps(dst + 11 * CH_DATA_BLK, zmm11);
            if (u_w > 12) _mm512_stream_ps(dst + 12 * CH_DATA_BLK, zmm12);
            if (u_w > 13) _mm512_stream_ps(dst + 13 * CH_DATA_BLK, zmm13);
            if (u_w > 14) _mm512_stream_ps(dst + 14 * CH_DATA_BLK, zmm14);
            if (u_w > 15) _mm512_stream_ps(dst + 15 * CH_DATA_BLK, zmm15);
            if (u_w > 16) _mm512_stream_ps(dst + 16 * CH_DATA_BLK, zmm16);
            if (u_w > 17) _mm512_stream_ps(dst + 17 * CH_DATA_BLK, zmm17);
            if (u_w > 18) _mm512_stream_ps(dst + 18 * CH_DATA_BLK, zmm18);
            if (u_w > 19) _mm512_stream_ps(dst + 19 * CH_DATA_BLK, zmm19);
            if (u_w > 20) _mm512_stream_ps(dst + 20 * CH_DATA_BLK, zmm20);
            if (u_w > 21) _mm512_stream_ps(dst + 21 * CH_DATA_BLK, zmm21);
            if (u_w > 22) _mm512_stream_ps(dst + 22 * CH_DATA_BLK, zmm22);
            if (u_w > 23) _mm512_stream_ps(dst + 23 * CH_DATA_BLK, zmm23);
            if (u_w > 24) _mm512_stream_ps(dst + 24 * CH_DATA_BLK, zmm24);
            if (u_w > 25) _mm512_stream_ps(dst + 25 * CH_DATA_BLK, zmm25);
            if (u_w > 26) _mm512_stream_ps(dst + 26 * CH_DATA_BLK, zmm26);
            if (u_w > 27) _mm512_stream_ps(dst + 27 * CH_DATA_BLK, zmm27);
            if (u_w > 28) _mm512_stream_ps(dst + 28 * CH_DATA_BLK, zmm28);
            if (u_w > 29) _mm512_stream_ps(dst + 29 * CH_DATA_BLK, zmm29);
            if (u_w > 30) _mm512_stream_ps(dst + 30 * CH_DATA_BLK, zmm30);
        } else {
            if (u_w > 0) _mm512_storeu_ps(dst + 0 * CH_DATA_BLK, zmm0);
            if (u_w > 1) _mm512_storeu_ps(dst + 1 * CH_DATA_BLK, zmm1);
            if (u_w > 2) _mm512_storeu_ps(dst + 2 * CH_DATA_BLK, zmm2);
            if (u_w > 3) _mm512_storeu_ps(dst + 3 * CH_DATA_BLK, zmm3);
            if (u_w > 4) _mm512_storeu_ps(dst + 4 * CH_DATA_BLK, zmm4);
            if (u_w > 5) _mm512_storeu_ps(dst + 5 * CH_DATA_BLK, zmm5);
            if (u_w > 6) _mm512_storeu_ps(dst + 6 * CH_DATA_BLK, zmm6);
            if (u_w > 7) _mm512_storeu_ps(dst + 7 * CH_DATA_BLK, zmm7);
            if (u_w > 8) _mm512_storeu_ps(dst + 8 * CH_DATA_BLK, zmm8);
            if (u_w > 9) _mm512_storeu_ps(dst + 9 * CH_DATA_BLK, zmm9);
            if (u_w > 10) _mm512_storeu_ps(dst + 10 * CH_DATA_BLK, zmm10);
            if (u_w > 11) _mm512_storeu_ps(dst + 11 * CH_DATA_BLK, zmm11);
            if (u_w > 12) _mm512_storeu_ps(dst + 12 * CH_DATA_BLK, zmm12);
            if (u_w > 13) _mm512_storeu_ps(dst + 13 * CH_DATA_BLK, zmm13);
            if (u_w > 14) _mm512_storeu_ps(dst + 14 * CH_DATA_BLK, zmm14);
            if (u_w > 15) _mm512_storeu_ps(dst + 15 * CH_DATA_BLK, zmm15);
            if (u_w > 16) _mm512_storeu_ps(dst + 16 * CH_DATA_BLK, zmm16);
            if (u_w > 17) _mm512_storeu_ps(dst + 17 * CH_DATA_BLK, zmm17);
            if (u_w > 18) _mm512_storeu_ps(dst + 18 * CH_DATA_BLK, zmm18);
            if (u_w > 19) _mm512_storeu_ps(dst + 19 * CH_DATA_BLK, zmm19);
            if (u_w > 20) _mm512_storeu_ps(dst + 20 * CH_DATA_BLK, zmm20);
            if (u_w > 21) _mm512_storeu_ps(dst + 21 * CH_DATA_BLK, zmm21);
            if (u_w > 22) _mm512_storeu_ps(dst + 22 * CH_DATA_BLK, zmm22);
            if (u_w > 23) _mm512_storeu_ps(dst + 23 * CH_DATA_BLK, zmm23);
            if (u_w > 24) _mm512_storeu_ps(dst + 24 * CH_DATA_BLK, zmm24);
            if (u_w > 25) _mm512_storeu_ps(dst + 25 * CH_DATA_BLK, zmm25);
            if (u_w > 26) _mm512_storeu_ps(dst + 26 * CH_DATA_BLK, zmm26);
            if (u_w > 27) _mm512_storeu_ps(dst + 27 * CH_DATA_BLK, zmm27);
            if (u_w > 28) _mm512_storeu_ps(dst + 28 * CH_DATA_BLK, zmm28);
            if (u_w > 29) _mm512_storeu_ps(dst + 29 * CH_DATA_BLK, zmm29);
            if (u_w > 30) _mm512_storeu_ps(dst + 30 * CH_DATA_BLK, zmm30);
        }
        dst += u_w * CH_DATA_BLK;
        dst_w -= u_w;
    } while (dst_w > 0);
    ker_p.pick<float*>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::DST_PTR_IDX) = dst;
#undef KW_COMPUTE_STEP
}

#define PD_CONV2D_DW_KERNEL_TABLE_BLK(NT_STORE, STRIDE_W) \
{\
    pd_conv2d_n16cx_depthwise_fp32_avx512_blk1x31_kernel<NT_STORE, STRIDE_W, 1>,\
    pd_conv2d_n16cx_depthwise_fp32_avx512_blk1x31_kernel<NT_STORE, STRIDE_W, 2>,\
    pd_conv2d_n16cx_depthwise_fp32_avx512_blk1x31_kernel<NT_STORE, STRIDE_W, 3>,\
    pd_conv2d_n16cx_depthwise_fp32_avx512_blk1x31_kernel<NT_STORE, STRIDE_W, 4>,\
    pd_conv2d_n16cx_depthwise_fp32_avx512_blk1x31_kernel<NT_STORE, STRIDE_W, 5>,\
    pd_conv2d_n16cx_depthwise_fp32_avx512_blk1x31_kernel<NT_STORE, STRIDE_W, 6>,\
    pd_conv2d_n16cx_depthwise_fp32_avx512_blk1x31_kernel<NT_STORE, STRIDE_W, 7>,\
    pd_conv2d_n16cx_depthwise_fp32_avx512_blk1x31_kernel<NT_STORE, STRIDE_W, 8>,\
    pd_conv2d_n16cx_depthwise_fp32_avx512_blk1x31_kernel<NT_STORE, STRIDE_W, 9>,\
    pd_conv2d_n16cx_depthwise_fp32_avx512_blk1x31_kernel<NT_STORE, STRIDE_W, 10>,\
    pd_conv2d_n16cx_depthwise_fp32_avx512_blk1x31_kernel<NT_STORE, STRIDE_W, 11>,\
    pd_conv2d_n16cx_depthwise_fp32_avx512_blk1x31_kernel<NT_STORE, STRIDE_W, 12>,\
    pd_conv2d_n16cx_depthwise_fp32_avx512_blk1x31_kernel<NT_STORE, STRIDE_W, 13>,\
    pd_conv2d_n16cx_depthwise_fp32_avx512_blk1x31_kernel<NT_STORE, STRIDE_W, 14>,\
}

const pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::func_t
    pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::table_[config::NT_STORE_OPT][config::SPEC_STRIDE_W_OPT][config::MAX_W_REGS] =
{
    {
        PD_CONV2D_DW_KERNEL_TABLE_BLK(false, 0),
        PD_CONV2D_DW_KERNEL_TABLE_BLK(false, 1),
        PD_CONV2D_DW_KERNEL_TABLE_BLK(false, 2),
    },
    {
        PD_CONV2D_DW_KERNEL_TABLE_BLK(true, 0),
        PD_CONV2D_DW_KERNEL_TABLE_BLK(true, 1),
        PD_CONV2D_DW_KERNEL_TABLE_BLK(true, 2),
    },
};

}}};
