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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_AVX512_CONV2D_N16CX_DIRECT_BLK1X9_KERNEL_FP32_AVX512_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_AVX512_CONV2D_N16CX_DIRECT_BLK1X9_KERNEL_FP32_AVX512_H_

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/direct/avx512/conv2d_n16cx_direct_kernel_fp32_avx512.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store, int32_t spec_stride_w, int32_t oc_len, int32_t w_len>
void conv2d_n16cx_direct_fp32_avx512_blk1x9_kernel(
    const int64_t *shar_param,
    int64_t *priv_param)
{
#define IC_COMPUTE_STEP(IC) do {\
    if (oc_len > 0 * CH_DT_BLK()) zmm27 = _mm512_loadu_ps(icb_flt_o16 + 0 * flt_ocb_stride + (IC) * CH_DT_BLK());\
    if (oc_len > 1 * CH_DT_BLK()) zmm28 = _mm512_loadu_ps(icb_flt_o16 + 1 * flt_ocb_stride + (IC) * CH_DT_BLK());\
    if (oc_len > 2 * CH_DT_BLK()) zmm29 = _mm512_loadu_ps(icb_flt_o48 + 0 * flt_ocb_stride + (IC) * CH_DT_BLK());\
    if (w_len > 0) {\
        zmm30 = _mm512_set1_ps(k_src[(IC) + 0 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm0  = _mm512_fmadd_ps(zmm27, zmm30, zmm0);\
        if (oc_len > 1 * CH_DT_BLK()) zmm9  = _mm512_fmadd_ps(zmm28, zmm30, zmm9);\
        if (oc_len > 2 * CH_DT_BLK()) zmm18 = _mm512_fmadd_ps(zmm29, zmm30, zmm18);\
    }\
    if (w_len > 1) {\
        zmm31 = _mm512_set1_ps(k_src[(IC) + 1 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm1  = _mm512_fmadd_ps(zmm27, zmm31, zmm1);\
        if (oc_len > 1 * CH_DT_BLK()) zmm10 = _mm512_fmadd_ps(zmm28, zmm31, zmm10);\
        if (oc_len > 2 * CH_DT_BLK()) zmm19 = _mm512_fmadd_ps(zmm29, zmm31, zmm19);\
    }\
    if (w_len > 2) {\
        zmm30 = _mm512_set1_ps(k_src[(IC) + 2 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm2  = _mm512_fmadd_ps(zmm27, zmm30, zmm2);\
        if (oc_len > 1 * CH_DT_BLK()) zmm11 = _mm512_fmadd_ps(zmm28, zmm30, zmm11);\
        if (oc_len > 2 * CH_DT_BLK()) zmm20 = _mm512_fmadd_ps(zmm29, zmm30, zmm20);\
    }\
    if (w_len > 3) {\
        zmm31 = _mm512_set1_ps(k_src[(IC) + 3 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm3  = _mm512_fmadd_ps(zmm27, zmm31, zmm3);\
        if (oc_len > 1 * CH_DT_BLK()) zmm12 = _mm512_fmadd_ps(zmm28, zmm31, zmm12);\
        if (oc_len > 2 * CH_DT_BLK()) zmm21 = _mm512_fmadd_ps(zmm29, zmm31, zmm21);\
    }\
    if (w_len > 4) {\
        zmm30 = _mm512_set1_ps(k_src[(IC) + 4 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm4  = _mm512_fmadd_ps(zmm27, zmm30, zmm4);\
        if (oc_len > 1 * CH_DT_BLK()) zmm13 = _mm512_fmadd_ps(zmm28, zmm30, zmm13);\
        if (oc_len > 2 * CH_DT_BLK()) zmm22 = _mm512_fmadd_ps(zmm29, zmm30, zmm22);\
    }\
    if (w_len > 5) {\
        zmm31 = _mm512_set1_ps(k_src[(IC) + 5 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm5  = _mm512_fmadd_ps(zmm27, zmm31, zmm5);\
        if (oc_len > 1 * CH_DT_BLK()) zmm14 = _mm512_fmadd_ps(zmm28, zmm31, zmm14);\
        if (oc_len > 2 * CH_DT_BLK()) zmm23 = _mm512_fmadd_ps(zmm29, zmm31, zmm23);\
    }\
    if (w_len > 6) {\
        zmm30 = _mm512_set1_ps(k_src[(IC) + 6 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm6  = _mm512_fmadd_ps(zmm27, zmm30, zmm6);\
        if (oc_len > 1 * CH_DT_BLK()) zmm15 = _mm512_fmadd_ps(zmm28, zmm30, zmm15);\
        if (oc_len > 2 * CH_DT_BLK()) zmm24 = _mm512_fmadd_ps(zmm29, zmm30, zmm24);\
    }\
    if (w_len > 7) {\
        zmm31 = _mm512_set1_ps(k_src[(IC) + 7 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm7  = _mm512_fmadd_ps(zmm27, zmm31, zmm7);\
        if (oc_len > 1 * CH_DT_BLK()) zmm16 = _mm512_fmadd_ps(zmm28, zmm31, zmm16);\
        if (oc_len > 2 * CH_DT_BLK()) zmm25 = _mm512_fmadd_ps(zmm29, zmm31, zmm25);\
    }\
    if (w_len > 8) {\
        zmm31 = _mm512_set1_ps(k_src[(IC) + 8 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm8  = _mm512_fmadd_ps(zmm27, zmm31, zmm8);\
        if (oc_len > 1 * CH_DT_BLK()) zmm17 = _mm512_fmadd_ps(zmm28, zmm31, zmm17);\
        if (oc_len > 2 * CH_DT_BLK()) zmm26 = _mm512_fmadd_ps(zmm29, zmm31, zmm26);\
    }\
} while (0)

#define IC_PREFETCH_STEP(IC)  do {\
    if (oc_len > 2 * CH_DT_BLK() && w_len > 4) {\
        if (oc_len > 0 * CH_DT_BLK()) _mm_prefetch((const char*)icb_flt_o16 + 0 * flt_ocb_stride + (IC) * CH_DT_BLK() + CH_DT_BLK() * CH_DT_BLK(), _MM_HINT_T0);\
        if (oc_len > 1 * CH_DT_BLK()) _mm_prefetch((const char*)icb_flt_o16 + 1 * flt_ocb_stride + (IC) * CH_DT_BLK() + CH_DT_BLK() * CH_DT_BLK(), _MM_HINT_T0);\
        if (oc_len > 2 * CH_DT_BLK()) _mm_prefetch((const char*)icb_flt_o48 + 0 * flt_ocb_stride + (IC) * CH_DT_BLK() + CH_DT_BLK() * CH_DT_BLK(), _MM_HINT_T0);\
    }\
} while (0)

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;
    __m512 zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;

    const int64_t kernel_h = shar_param[KH_IDX()];
    const int64_t kernel_w = shar_param[KW_IDX()];
    const int64_t src_sw_stride = spec_stride_w ? spec_stride_w * CH_DT_BLK() : shar_param[SRC_SW_STRIDE_IDX()];
    const int64_t src_dw_stride = shar_param[SRC_DW_STRIDE_IDX()];
    const int64_t src_dh_stride = shar_param[SRC_DH_STRIDE_IDX()] - kernel_w * src_dw_stride;
    const int64_t flt_icb_stride = (kernel_h - priv_param[KH_END_IDX()] + priv_param[KH_START_IDX()]) * kernel_w * CH_DT_BLK() * CH_DT_BLK();
    const int64_t src_icb_stride = shar_param[SRC_ICB_STRIDE_IDX()];

    const int64_t src_offset = priv_param[KH_START_IDX()] * shar_param[SRC_DH_STRIDE_IDX()];
    const int64_t flt_offset = priv_param[KH_START_IDX()] * kernel_w * CH_DT_BLK() * CH_DT_BLK();

    const float *src = PICK_PARAM(const float*, priv_param, SRC_IDX());
    const float *his = PICK_PARAM(const float*, priv_param, HIS_IDX());
    float *dst       = PICK_PARAM(float*, priv_param, DST_IDX());
    int64_t ow       = priv_param[OW_IDX()];
    do {
        const int64_t kernel_flags = shar_param[FLAGS_IDX()];
        if (kernel_flags & KERNEL_FLAG_LD_BIAS()) {
            const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
            if (oc_len > 0 * CH_DT_BLK()) {
                if (w_len > 0) zmm0 = _mm512_loadu_ps(bias + 0 * CH_DT_BLK());
                if (w_len > 1) zmm1 = zmm0;
                if (w_len > 2) zmm2 = zmm0;
                if (w_len > 3) zmm3 = zmm0;
                if (w_len > 4) zmm4 = zmm0;
                if (w_len > 5) zmm5 = zmm0;
                if (w_len > 6) zmm6 = zmm0;
                if (w_len > 7) zmm7 = zmm0;
                if (w_len > 8) zmm8 = zmm0;
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                if (w_len > 0) zmm9  = _mm512_loadu_ps(bias + 1 * CH_DT_BLK());
                if (w_len > 1) zmm10 = zmm9;
                if (w_len > 2) zmm11 = zmm9;
                if (w_len > 3) zmm12 = zmm9;
                if (w_len > 4) zmm13 = zmm9;
                if (w_len > 5) zmm14 = zmm9;
                if (w_len > 6) zmm15 = zmm9;
                if (w_len > 7) zmm16 = zmm9;
                if (w_len > 8) zmm17 = zmm9;
            }
            if (oc_len > 2 * CH_DT_BLK()) {
                if (w_len > 0) zmm18 = _mm512_loadu_ps(bias + 2 * CH_DT_BLK());
                if (w_len > 1) zmm19 = zmm18;
                if (w_len > 2) zmm20 = zmm18;
                if (w_len > 3) zmm21 = zmm18;
                if (w_len > 4) zmm22 = zmm18;
                if (w_len > 5) zmm23 = zmm18;
                if (w_len > 6) zmm24 = zmm18;
                if (w_len > 7) zmm25 = zmm18;
                if (w_len > 8) zmm26 = zmm18;
            }
        } else {
            const float *l_his = his;
            const int64_t his_ocb_stride = shar_param[HIS_OCB_STRIDE_IDX()];
            if (oc_len > 0 * CH_DT_BLK()) {
                if (w_len > 0) zmm0 = _mm512_loadu_ps(l_his + 0 * CH_DT_BLK());
                if (w_len > 1) zmm1 = _mm512_loadu_ps(l_his + 1 * CH_DT_BLK());
                if (w_len > 2) zmm2 = _mm512_loadu_ps(l_his + 2 * CH_DT_BLK());
                if (w_len > 3) zmm3 = _mm512_loadu_ps(l_his + 3 * CH_DT_BLK());
                if (w_len > 4) zmm4 = _mm512_loadu_ps(l_his + 4 * CH_DT_BLK());
                if (w_len > 5) zmm5 = _mm512_loadu_ps(l_his + 5 * CH_DT_BLK());
                if (w_len > 6) zmm6 = _mm512_loadu_ps(l_his + 6 * CH_DT_BLK());
                if (w_len > 7) zmm7 = _mm512_loadu_ps(l_his + 7 * CH_DT_BLK());
                if (w_len > 8) zmm8 = _mm512_loadu_ps(l_his + 8 * CH_DT_BLK());
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                l_his += his_ocb_stride;
                if (w_len > 0) zmm9  = _mm512_loadu_ps(l_his + 0 * CH_DT_BLK());
                if (w_len > 1) zmm10 = _mm512_loadu_ps(l_his + 1 * CH_DT_BLK());
                if (w_len > 2) zmm11 = _mm512_loadu_ps(l_his + 2 * CH_DT_BLK());
                if (w_len > 3) zmm12 = _mm512_loadu_ps(l_his + 3 * CH_DT_BLK());
                if (w_len > 4) zmm13 = _mm512_loadu_ps(l_his + 4 * CH_DT_BLK());
                if (w_len > 5) zmm14 = _mm512_loadu_ps(l_his + 5 * CH_DT_BLK());
                if (w_len > 6) zmm15 = _mm512_loadu_ps(l_his + 6 * CH_DT_BLK());
                if (w_len > 7) zmm16 = _mm512_loadu_ps(l_his + 7 * CH_DT_BLK());
                if (w_len > 8) zmm17 = _mm512_loadu_ps(l_his + 8 * CH_DT_BLK());
            }
            if (oc_len > 2 * CH_DT_BLK()) {
                l_his += his_ocb_stride;
                if (w_len > 0) zmm18 = _mm512_loadu_ps(l_his + 0 * CH_DT_BLK());
                if (w_len > 1) zmm19 = _mm512_loadu_ps(l_his + 1 * CH_DT_BLK());
                if (w_len > 2) zmm20 = _mm512_loadu_ps(l_his + 2 * CH_DT_BLK());
                if (w_len > 3) zmm21 = _mm512_loadu_ps(l_his + 3 * CH_DT_BLK());
                if (w_len > 4) zmm22 = _mm512_loadu_ps(l_his + 4 * CH_DT_BLK());
                if (w_len > 5) zmm23 = _mm512_loadu_ps(l_his + 5 * CH_DT_BLK());
                if (w_len > 6) zmm24 = _mm512_loadu_ps(l_his + 6 * CH_DT_BLK());
                if (w_len > 7) zmm25 = _mm512_loadu_ps(l_his + 7 * CH_DT_BLK());
                if (w_len > 8) zmm26 = _mm512_loadu_ps(l_his + 8 * CH_DT_BLK());
            }
        }

        if (kernel_flags & KERNEL_FLAG_AD_BIAS()) {
            const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
            if (oc_len > 0 * CH_DT_BLK()) {
                zmm30 = _mm512_loadu_ps(bias + 0 * CH_DT_BLK());
                if (w_len > 0) zmm0 = _mm512_add_ps(zmm30, zmm0);
                if (w_len > 1) zmm1 = _mm512_add_ps(zmm30, zmm1);
                if (w_len > 2) zmm2 = _mm512_add_ps(zmm30, zmm2);
                if (w_len > 3) zmm3 = _mm512_add_ps(zmm30, zmm3);
                if (w_len > 4) zmm4 = _mm512_add_ps(zmm30, zmm4);
                if (w_len > 5) zmm5 = _mm512_add_ps(zmm30, zmm5);
                if (w_len > 6) zmm6 = _mm512_add_ps(zmm30, zmm6);
                if (w_len > 7) zmm7 = _mm512_add_ps(zmm30, zmm7);
                if (w_len > 8) zmm8 = _mm512_add_ps(zmm30, zmm8);
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                zmm31 = _mm512_loadu_ps(bias + 1 * CH_DT_BLK());
                if (w_len > 0) zmm9  = _mm512_add_ps(zmm31, zmm9);
                if (w_len > 1) zmm10 = _mm512_add_ps(zmm31, zmm10);
                if (w_len > 2) zmm11 = _mm512_add_ps(zmm31, zmm11);
                if (w_len > 3) zmm12 = _mm512_add_ps(zmm31, zmm12);
                if (w_len > 4) zmm13 = _mm512_add_ps(zmm31, zmm13);
                if (w_len > 5) zmm14 = _mm512_add_ps(zmm31, zmm14);
                if (w_len > 6) zmm15 = _mm512_add_ps(zmm31, zmm15);
                if (w_len > 7) zmm16 = _mm512_add_ps(zmm31, zmm16);
                if (w_len > 8) zmm17 = _mm512_add_ps(zmm31, zmm17);
            }
            if (oc_len > 2 * CH_DT_BLK()) {
                zmm30 = _mm512_loadu_ps(bias + 2 * CH_DT_BLK());
                if (w_len > 0) zmm18 = _mm512_add_ps(zmm30, zmm18);
                if (w_len > 1) zmm19 = _mm512_add_ps(zmm30, zmm19);
                if (w_len > 2) zmm20 = _mm512_add_ps(zmm30, zmm20);
                if (w_len > 3) zmm21 = _mm512_add_ps(zmm30, zmm21);
                if (w_len > 4) zmm22 = _mm512_add_ps(zmm30, zmm22);
                if (w_len > 5) zmm23 = _mm512_add_ps(zmm30, zmm23);
                if (w_len > 6) zmm24 = _mm512_add_ps(zmm30, zmm24);
                if (w_len > 7) zmm25 = _mm512_add_ps(zmm30, zmm25);
                if (w_len > 8) zmm26 = _mm512_add_ps(zmm30, zmm26);
            }
        }
        
        const int64_t flt_ocb_stride = shar_param[FLT_OCB_STRIDE_IDX()];
        const int64_t kh_start = priv_param[KH_START_IDX()];
        const int64_t kh_end = priv_param[KH_END_IDX()];
        int64_t channels     = shar_param[CHANNELS_IDX()];
        const float *icb_src = src + src_offset;
        const float *icb_flt_o16 = PICK_PARAM(const float*, priv_param, FLT_IDX()) + flt_offset;
        const float *icb_flt_o48 = icb_flt_o16 + 2 * flt_ocb_stride;
        while (channels >= CH_DT_BLK()) {
            channels -= CH_DT_BLK();
            const float *k_src = icb_src;
            for (int64_t kh = kh_start; kh < kh_end; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    for (int64_t ic = 0; ic < CH_DT_BLK(); ic += 4) {
                        IC_COMPUTE_STEP(0);
                        IC_PREFETCH_STEP(0);
                        IC_COMPUTE_STEP(1);
                        IC_PREFETCH_STEP(1);
                        IC_COMPUTE_STEP(2);
                        IC_PREFETCH_STEP(2);
                        IC_COMPUTE_STEP(3);
                        IC_PREFETCH_STEP(3);
                        k_src += 4;
                        icb_flt_o16 += 4 * CH_DT_BLK();
                        icb_flt_o48 += 4 * CH_DT_BLK();
                    }
                    k_src += src_dw_stride - CH_DT_BLK();
                }
                k_src += src_dh_stride;
            }
            icb_flt_o16 += flt_icb_stride;
            icb_flt_o48 += flt_icb_stride;
            icb_src += src_icb_stride;
        }
        if (channels > 0) {
            const float *k_src = icb_src;
            for (int64_t kh = kh_start; kh < kh_end; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    for (int64_t ic = 0; ic < channels; ++ic) {
                        IC_COMPUTE_STEP(0);
                        k_src += 1;
                        icb_flt_o16 += CH_DT_BLK();
                        icb_flt_o48 += CH_DT_BLK();
                    }
                    icb_flt_o16 += (CH_DT_BLK() - channels) * CH_DT_BLK();
                    icb_flt_o48 += (CH_DT_BLK() - channels) * CH_DT_BLK();
                    k_src += src_dw_stride - channels;
                }
                k_src += src_dh_stride;
            }
        }
        
        if (kernel_flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
            zmm30 = _mm512_setzero_ps();
            if (oc_len > 0 * CH_DT_BLK()) {
                if (w_len > 0) zmm0 = _mm512_max_ps(zmm0, zmm30);
                if (w_len > 1) zmm1 = _mm512_max_ps(zmm1, zmm30);
                if (w_len > 2) zmm2 = _mm512_max_ps(zmm2, zmm30);
                if (w_len > 3) zmm3 = _mm512_max_ps(zmm3, zmm30);
                if (w_len > 4) zmm4 = _mm512_max_ps(zmm4, zmm30);
                if (w_len > 5) zmm5 = _mm512_max_ps(zmm5, zmm30);
                if (w_len > 6) zmm6 = _mm512_max_ps(zmm6, zmm30);
                if (w_len > 7) zmm7 = _mm512_max_ps(zmm7, zmm30);
                if (w_len > 8) zmm8 = _mm512_max_ps(zmm8, zmm30);
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                if (w_len > 0) zmm9  = _mm512_max_ps(zmm9, zmm30);
                if (w_len > 1) zmm10 = _mm512_max_ps(zmm10, zmm30);
                if (w_len > 2) zmm11 = _mm512_max_ps(zmm11, zmm30);
                if (w_len > 3) zmm12 = _mm512_max_ps(zmm12, zmm30);
                if (w_len > 4) zmm13 = _mm512_max_ps(zmm13, zmm30);
                if (w_len > 5) zmm14 = _mm512_max_ps(zmm14, zmm30);
                if (w_len > 6) zmm15 = _mm512_max_ps(zmm15, zmm30);
                if (w_len > 7) zmm16 = _mm512_max_ps(zmm16, zmm30);
                if (w_len > 8) zmm17 = _mm512_max_ps(zmm17, zmm30);
            }
            if (oc_len > 2 * CH_DT_BLK()) {
                if (w_len > 0) zmm18 = _mm512_max_ps(zmm18, zmm30);
                if (w_len > 1) zmm19 = _mm512_max_ps(zmm19, zmm30);
                if (w_len > 2) zmm20 = _mm512_max_ps(zmm20, zmm30);
                if (w_len > 3) zmm21 = _mm512_max_ps(zmm21, zmm30);
                if (w_len > 4) zmm22 = _mm512_max_ps(zmm22, zmm30);
                if (w_len > 5) zmm23 = _mm512_max_ps(zmm23, zmm30);
                if (w_len > 6) zmm24 = _mm512_max_ps(zmm24, zmm30);
                if (w_len > 7) zmm25 = _mm512_max_ps(zmm25, zmm30);
                if (w_len > 8) zmm26 = _mm512_max_ps(zmm26, zmm30);
            }
        }
        if (kernel_flags & KERNEL_FLAG_RELU6()) {
            zmm31 = _mm512_set1_ps(6.0f);
            if (oc_len > 0 * CH_DT_BLK()) {
                if (w_len > 0) zmm0 = _mm512_min_ps(zmm0, zmm31);
                if (w_len > 1) zmm1 = _mm512_min_ps(zmm1, zmm31);
                if (w_len > 2) zmm2 = _mm512_min_ps(zmm2, zmm31);
                if (w_len > 3) zmm3 = _mm512_min_ps(zmm3, zmm31);
                if (w_len > 4) zmm4 = _mm512_min_ps(zmm4, zmm31);
                if (w_len > 5) zmm5 = _mm512_min_ps(zmm5, zmm31);
                if (w_len > 6) zmm6 = _mm512_min_ps(zmm6, zmm31);
                if (w_len > 7) zmm7 = _mm512_min_ps(zmm7, zmm31);
                if (w_len > 8) zmm8 = _mm512_min_ps(zmm8, zmm31);
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                if (w_len > 0) zmm9  = _mm512_min_ps(zmm9, zmm31);
                if (w_len > 1) zmm10 = _mm512_min_ps(zmm10, zmm31);
                if (w_len > 2) zmm11 = _mm512_min_ps(zmm11, zmm31);
                if (w_len > 3) zmm12 = _mm512_min_ps(zmm12, zmm31);
                if (w_len > 4) zmm13 = _mm512_min_ps(zmm13, zmm31);
                if (w_len > 5) zmm14 = _mm512_min_ps(zmm14, zmm31);
                if (w_len > 6) zmm15 = _mm512_min_ps(zmm15, zmm31);
                if (w_len > 7) zmm16 = _mm512_min_ps(zmm16, zmm31);
                if (w_len > 8) zmm17 = _mm512_min_ps(zmm17, zmm31);
            }
            if (oc_len > 2 * CH_DT_BLK()) {
                if (w_len > 0) zmm18 = _mm512_min_ps(zmm18, zmm31);
                if (w_len > 1) zmm19 = _mm512_min_ps(zmm19, zmm31);
                if (w_len > 2) zmm20 = _mm512_min_ps(zmm20, zmm31);
                if (w_len > 3) zmm21 = _mm512_min_ps(zmm21, zmm31);
                if (w_len > 4) zmm22 = _mm512_min_ps(zmm22, zmm31);
                if (w_len > 5) zmm23 = _mm512_min_ps(zmm23, zmm31);
                if (w_len > 6) zmm24 = _mm512_min_ps(zmm24, zmm31);
                if (w_len > 5) zmm25 = _mm512_min_ps(zmm25, zmm31);
                if (w_len > 6) zmm26 = _mm512_min_ps(zmm26, zmm31);
            }
        }

        if (nt_store) {
            float* l_dst = dst;
            const int64_t dst_ocb_stride = shar_param[DST_OCB_STRIDE_IDX()];
            if (oc_len > 0 * CH_DT_BLK()) {
                if (w_len > 0) _mm512_stream_ps(l_dst + 0 * CH_DT_BLK(), zmm0);
                if (w_len > 1) _mm512_stream_ps(l_dst + 1 * CH_DT_BLK(), zmm1);
                if (w_len > 2) _mm512_stream_ps(l_dst + 2 * CH_DT_BLK(), zmm2);
                if (w_len > 3) _mm512_stream_ps(l_dst + 3 * CH_DT_BLK(), zmm3);
                if (w_len > 4) _mm512_stream_ps(l_dst + 4 * CH_DT_BLK(), zmm4);
                if (w_len > 5) _mm512_stream_ps(l_dst + 5 * CH_DT_BLK(), zmm5);
                if (w_len > 6) _mm512_stream_ps(l_dst + 6 * CH_DT_BLK(), zmm6);
                if (w_len > 7) _mm512_stream_ps(l_dst + 7 * CH_DT_BLK(), zmm7);
                if (w_len > 8) _mm512_stream_ps(l_dst + 8 * CH_DT_BLK(), zmm8);
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                if (w_len > 0) _mm512_stream_ps(l_dst + 0 * CH_DT_BLK(), zmm9);
                if (w_len > 1) _mm512_stream_ps(l_dst + 1 * CH_DT_BLK(), zmm10);
                if (w_len > 2) _mm512_stream_ps(l_dst + 2 * CH_DT_BLK(), zmm11);
                if (w_len > 3) _mm512_stream_ps(l_dst + 3 * CH_DT_BLK(), zmm12);
                if (w_len > 4) _mm512_stream_ps(l_dst + 4 * CH_DT_BLK(), zmm13);
                if (w_len > 5) _mm512_stream_ps(l_dst + 5 * CH_DT_BLK(), zmm14);
                if (w_len > 6) _mm512_stream_ps(l_dst + 6 * CH_DT_BLK(), zmm15);
                if (w_len > 7) _mm512_stream_ps(l_dst + 7 * CH_DT_BLK(), zmm16);
                if (w_len > 8) _mm512_stream_ps(l_dst + 8 * CH_DT_BLK(), zmm17);
            }
            if (oc_len > 2 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                if (w_len > 0) _mm512_stream_ps(l_dst + 0 * CH_DT_BLK(), zmm18);
                if (w_len > 1) _mm512_stream_ps(l_dst + 1 * CH_DT_BLK(), zmm19);
                if (w_len > 2) _mm512_stream_ps(l_dst + 2 * CH_DT_BLK(), zmm20);
                if (w_len > 3) _mm512_stream_ps(l_dst + 3 * CH_DT_BLK(), zmm21);
                if (w_len > 4) _mm512_stream_ps(l_dst + 4 * CH_DT_BLK(), zmm22);
                if (w_len > 5) _mm512_stream_ps(l_dst + 5 * CH_DT_BLK(), zmm23);
                if (w_len > 6) _mm512_stream_ps(l_dst + 6 * CH_DT_BLK(), zmm24);
                if (w_len > 7) _mm512_stream_ps(l_dst + 7 * CH_DT_BLK(), zmm25);
                if (w_len > 8) _mm512_stream_ps(l_dst + 8 * CH_DT_BLK(), zmm26);
            }
        } else {
            float* l_dst = dst;
            const int64_t dst_ocb_stride = shar_param[DST_OCB_STRIDE_IDX()];
            if (oc_len > 0 * CH_DT_BLK()) {
                if (w_len > 0) _mm512_storeu_ps(l_dst + 0 * CH_DT_BLK(), zmm0);
                if (w_len > 1) _mm512_storeu_ps(l_dst + 1 * CH_DT_BLK(), zmm1);
                if (w_len > 2) _mm512_storeu_ps(l_dst + 2 * CH_DT_BLK(), zmm2);
                if (w_len > 3) _mm512_storeu_ps(l_dst + 3 * CH_DT_BLK(), zmm3);
                if (w_len > 4) _mm512_storeu_ps(l_dst + 4 * CH_DT_BLK(), zmm4);
                if (w_len > 5) _mm512_storeu_ps(l_dst + 5 * CH_DT_BLK(), zmm5);
                if (w_len > 6) _mm512_storeu_ps(l_dst + 6 * CH_DT_BLK(), zmm6);
                if (w_len > 7) _mm512_storeu_ps(l_dst + 7 * CH_DT_BLK(), zmm7);
                if (w_len > 8) _mm512_storeu_ps(l_dst + 8 * CH_DT_BLK(), zmm8);
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                if (w_len > 0) _mm512_storeu_ps(l_dst + 0 * CH_DT_BLK(), zmm9);
                if (w_len > 1) _mm512_storeu_ps(l_dst + 1 * CH_DT_BLK(), zmm10);
                if (w_len > 2) _mm512_storeu_ps(l_dst + 2 * CH_DT_BLK(), zmm11);
                if (w_len > 3) _mm512_storeu_ps(l_dst + 3 * CH_DT_BLK(), zmm12);
                if (w_len > 4) _mm512_storeu_ps(l_dst + 4 * CH_DT_BLK(), zmm13);
                if (w_len > 5) _mm512_storeu_ps(l_dst + 5 * CH_DT_BLK(), zmm14);
                if (w_len > 6) _mm512_storeu_ps(l_dst + 6 * CH_DT_BLK(), zmm15);
                if (w_len > 7) _mm512_storeu_ps(l_dst + 7 * CH_DT_BLK(), zmm16);
                if (w_len > 8) _mm512_storeu_ps(l_dst + 8 * CH_DT_BLK(), zmm17);
            }
            if (oc_len > 2 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                if (w_len > 0) _mm512_storeu_ps(l_dst + 0 * CH_DT_BLK(), zmm18);
                if (w_len > 1) _mm512_storeu_ps(l_dst + 1 * CH_DT_BLK(), zmm19);
                if (w_len > 2) _mm512_storeu_ps(l_dst + 2 * CH_DT_BLK(), zmm20);
                if (w_len > 3) _mm512_storeu_ps(l_dst + 3 * CH_DT_BLK(), zmm21);
                if (w_len > 4) _mm512_storeu_ps(l_dst + 4 * CH_DT_BLK(), zmm22);
                if (w_len > 5) _mm512_storeu_ps(l_dst + 5 * CH_DT_BLK(), zmm23);
                if (w_len > 6) _mm512_storeu_ps(l_dst + 6 * CH_DT_BLK(), zmm24);
                if (w_len > 7) _mm512_storeu_ps(l_dst + 7 * CH_DT_BLK(), zmm25);
                if (w_len > 8) _mm512_storeu_ps(l_dst + 8 * CH_DT_BLK(), zmm26);
            }
        }
        src += w_len * src_sw_stride;
        his += w_len * CH_DT_BLK();
        dst += w_len * CH_DT_BLK();
        ow -= w_len;
    } while (ow > 0);
    PICK_PARAM(const float *, priv_param, SRC_IDX()) = src;
    PICK_PARAM(const float *, priv_param, HIS_IDX()) = his;
    PICK_PARAM(float *, priv_param, DST_IDX()) = dst;
#undef IC_COMPUTE_STEP
#undef IC_PREFETCH_STEP
}

}}};

#endif
