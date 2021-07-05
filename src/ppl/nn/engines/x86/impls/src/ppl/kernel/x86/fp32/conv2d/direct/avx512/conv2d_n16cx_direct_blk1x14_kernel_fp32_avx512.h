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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_AVX512_CONV2D_N16CX_DIRECT_BLK1X14_KERNEL_FP32_AVX512_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_AVX512_CONV2D_N16CX_DIRECT_BLK1X14_KERNEL_FP32_AVX512_H_

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/direct/avx512/conv2d_n16cx_direct_kernel_fp32_avx512.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store, int32_t spec_stride_w, int32_t oc_len, int32_t w_len>
void conv2d_n16cx_direct_fp32_avx512_blk1x14_kernel(
    const int64_t *priv_param,
    const int64_t *shar_param)
{
#define IC_COMPUTE_STEP(IC) do {\
    if (oc_len > 0 * CH_DT_BLK()) zmm28 = _mm512_loadu_ps(ic_flt + 0 * flt_ocb_stride + (IC) * CH_DT_BLK());\
    if (oc_len > 1 * CH_DT_BLK()) zmm29 = _mm512_loadu_ps(ic_flt + 1 * flt_ocb_stride + (IC) * CH_DT_BLK());\
    if (w_len > 0) {\
        zmm30 = _mm512_set1_ps(ic_src[(IC) + 0 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm0  = _mm512_fmadd_ps(zmm28, zmm30, zmm0);\
        if (oc_len > 1 * CH_DT_BLK()) zmm14 = _mm512_fmadd_ps(zmm29, zmm30, zmm14);\
    }\
    if (w_len > 1) {\
        zmm31 = _mm512_set1_ps(ic_src[(IC) + 1 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm1  = _mm512_fmadd_ps(zmm28, zmm31, zmm1);\
        if (oc_len > 1 * CH_DT_BLK()) zmm15 = _mm512_fmadd_ps(zmm29, zmm31, zmm15);\
    }\
    if (w_len > 2) {\
        zmm30 = _mm512_set1_ps(ic_src[(IC) + 2 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm2  = _mm512_fmadd_ps(zmm28, zmm30, zmm2);\
        if (oc_len > 1 * CH_DT_BLK()) zmm16 = _mm512_fmadd_ps(zmm29, zmm30, zmm16);\
    }\
    if (w_len > 3) {\
        zmm31 = _mm512_set1_ps(ic_src[(IC) + 3 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm3  = _mm512_fmadd_ps(zmm28, zmm31, zmm3);\
        if (oc_len > 1 * CH_DT_BLK()) zmm17 = _mm512_fmadd_ps(zmm29, zmm31, zmm17);\
    }\
    if (w_len > 4) {\
        zmm30 = _mm512_set1_ps(ic_src[(IC) + 4 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm4  = _mm512_fmadd_ps(zmm28, zmm30, zmm4);\
        if (oc_len > 1 * CH_DT_BLK()) zmm18 = _mm512_fmadd_ps(zmm29, zmm30, zmm18);\
    }\
    if (w_len > 5) {\
        zmm31 = _mm512_set1_ps(ic_src[(IC) + 5 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm5  = _mm512_fmadd_ps(zmm28, zmm31, zmm5);\
        if (oc_len > 1 * CH_DT_BLK()) zmm19 = _mm512_fmadd_ps(zmm29, zmm31, zmm19);\
    }\
    if (w_len > 6) {\
        zmm30 = _mm512_set1_ps(ic_src[(IC) + 6 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm6  = _mm512_fmadd_ps(zmm28, zmm30, zmm6);\
        if (oc_len > 1 * CH_DT_BLK()) zmm20 = _mm512_fmadd_ps(zmm29, zmm30, zmm20);\
    }\
    if (w_len > 6) {\
        zmm31 = _mm512_set1_ps(ic_src[(IC) + 7 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm7  = _mm512_fmadd_ps(zmm28, zmm31, zmm7);\
        if (oc_len > 1 * CH_DT_BLK()) zmm21 = _mm512_fmadd_ps(zmm29, zmm31, zmm21);\
    }\
    if (w_len > 8) {\
        zmm30 = _mm512_set1_ps(ic_src[(IC) + 8 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm8  = _mm512_fmadd_ps(zmm28, zmm30, zmm8);\
        if (oc_len > 1 * CH_DT_BLK()) zmm22 = _mm512_fmadd_ps(zmm29, zmm30, zmm22);\
    }\
    if (w_len > 9) {\
        zmm31 = _mm512_set1_ps(ic_src[(IC) + 9 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm9  = _mm512_fmadd_ps(zmm28, zmm31, zmm9);\
        if (oc_len > 1 * CH_DT_BLK()) zmm23 = _mm512_fmadd_ps(zmm29, zmm31, zmm23);\
    }\
    if (w_len > 10) {\
        zmm30 = _mm512_set1_ps(ic_src[(IC) + 10 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm10 = _mm512_fmadd_ps(zmm28, zmm30, zmm10);\
        if (oc_len > 1 * CH_DT_BLK()) zmm24 = _mm512_fmadd_ps(zmm29, zmm30, zmm24);\
    }\
    if (w_len > 11) {\
        zmm31 = _mm512_set1_ps(ic_src[(IC) + 11 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm11 = _mm512_fmadd_ps(zmm28, zmm31, zmm11);\
        if (oc_len > 1 * CH_DT_BLK()) zmm25 = _mm512_fmadd_ps(zmm29, zmm31, zmm25);\
    }\
    if (w_len > 12) {\
        zmm30 = _mm512_set1_ps(ic_src[(IC) + 12 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm12 = _mm512_fmadd_ps(zmm28, zmm30, zmm12);\
        if (oc_len > 1 * CH_DT_BLK()) zmm26 = _mm512_fmadd_ps(zmm29, zmm30, zmm26);\
    }\
    if (w_len > 13) {\
        zmm31 = _mm512_set1_ps(ic_src[(IC) + 13 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm13 = _mm512_fmadd_ps(zmm28, zmm31, zmm13);\
        if (oc_len > 1 * CH_DT_BLK()) zmm27 = _mm512_fmadd_ps(zmm29, zmm31, zmm27);\
    }\
} while (0)

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;
    __m512 zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;

    const int64_t kernel_h = shar_param[KH_IDX()];
    const int64_t kernel_w = shar_param[KW_IDX()];
    const int64_t src_icb_stride = shar_param[SRC_ICB_STRIDE_IDX()];
    const int64_t src_sw_stride = spec_stride_w ? spec_stride_w * CH_DT_BLK() : shar_param[SRC_SW_STRIDE_IDX()];
    const int64_t src_dw_stride = shar_param[SRC_DW_STRIDE_IDX()];
    const int64_t src_dh_stride = shar_param[SRC_DH_STRIDE_IDX()] - kernel_w * src_dw_stride;
    const int64_t flt_ocb_stride = shar_param[FLT_OCB_STRIDE_IDX()];
    const int64_t kernel_flags = shar_param[FLAGS_IDX()];
    const int64_t kh_start = priv_param[KH_START_IDX()];
    const int64_t kh_end = priv_param[KH_END_IDX()];

    const float *src = PICK_PARAM(const float*, priv_param, SRC_IDX());
    const float *his = PICK_PARAM(const float*, priv_param, HIS_IDX());
    float *dst       = PICK_PARAM(float*, priv_param, DST_IDX());
    int64_t ow       = priv_param[OW_IDX()];
    do {
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
                if (w_len > 9) zmm9 = zmm0;
                if (w_len > 10) zmm10 = zmm0;
                if (w_len > 11) zmm11 = zmm0;
                if (w_len > 12) zmm12 = zmm0;
                if (w_len > 13) zmm13 = zmm0;
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                if (w_len > 0) zmm14 = _mm512_loadu_ps(bias + 1 * CH_DT_BLK());
                if (w_len > 1) zmm15 = zmm14;
                if (w_len > 2) zmm16 = zmm14;
                if (w_len > 3) zmm17 = zmm14;
                if (w_len > 4) zmm18 = zmm14;
                if (w_len > 5) zmm19 = zmm14;
                if (w_len > 6) zmm20 = zmm14;
                if (w_len > 7) zmm21 = zmm14;
                if (w_len > 8) zmm22 = zmm14;
                if (w_len > 9) zmm23 = zmm14;
                if (w_len > 10) zmm24 = zmm14;
                if (w_len > 11) zmm25 = zmm14;
                if (w_len > 12) zmm26 = zmm14;
                if (w_len > 13) zmm27 = zmm14;
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
                if (w_len > 9) zmm9 = _mm512_loadu_ps(l_his + 9 * CH_DT_BLK());
                if (w_len > 10) zmm10 = _mm512_loadu_ps(l_his + 10 * CH_DT_BLK());
                if (w_len > 11) zmm11 = _mm512_loadu_ps(l_his + 11 * CH_DT_BLK());
                if (w_len > 12) zmm12 = _mm512_loadu_ps(l_his + 12 * CH_DT_BLK());
                if (w_len > 13) zmm13 = _mm512_loadu_ps(l_his + 13 * CH_DT_BLK());
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                l_his += his_ocb_stride;
                if (w_len > 0) zmm14 = _mm512_loadu_ps(l_his + 0 * CH_DT_BLK());
                if (w_len > 1) zmm15 = _mm512_loadu_ps(l_his + 1 * CH_DT_BLK());
                if (w_len > 2) zmm16 = _mm512_loadu_ps(l_his + 2 * CH_DT_BLK());
                if (w_len > 3) zmm17 = _mm512_loadu_ps(l_his + 3 * CH_DT_BLK());
                if (w_len > 4) zmm18 = _mm512_loadu_ps(l_his + 4 * CH_DT_BLK());
                if (w_len > 5) zmm19 = _mm512_loadu_ps(l_his + 5 * CH_DT_BLK());
                if (w_len > 6) zmm20 = _mm512_loadu_ps(l_his + 6 * CH_DT_BLK());
                if (w_len > 7) zmm21 = _mm512_loadu_ps(l_his + 7 * CH_DT_BLK());
                if (w_len > 8) zmm22 = _mm512_loadu_ps(l_his + 8 * CH_DT_BLK());
                if (w_len > 9) zmm23 = _mm512_loadu_ps(l_his + 9 * CH_DT_BLK());
                if (w_len > 10) zmm24 = _mm512_loadu_ps(l_his + 10 * CH_DT_BLK());
                if (w_len > 11) zmm25 = _mm512_loadu_ps(l_his + 11 * CH_DT_BLK());
                if (w_len > 12) zmm26 = _mm512_loadu_ps(l_his + 12 * CH_DT_BLK());
                if (w_len > 13) zmm27 = _mm512_loadu_ps(l_his + 13 * CH_DT_BLK());
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
                if (w_len > 9) zmm9 = _mm512_add_ps(zmm30, zmm9);
                if (w_len > 10) zmm10 = _mm512_add_ps(zmm30, zmm10);
                if (w_len > 11) zmm11 = _mm512_add_ps(zmm30, zmm11);
                if (w_len > 12) zmm12 = _mm512_add_ps(zmm30, zmm12);
                if (w_len > 13) zmm13 = _mm512_add_ps(zmm30, zmm13);
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                zmm31 = _mm512_loadu_ps(bias + 1 * CH_DT_BLK());
                if (w_len > 0) zmm14 = _mm512_add_ps(zmm31, zmm14);
                if (w_len > 1) zmm15 = _mm512_add_ps(zmm31, zmm15);
                if (w_len > 2) zmm16 = _mm512_add_ps(zmm31, zmm16);
                if (w_len > 3) zmm17 = _mm512_add_ps(zmm31, zmm17);
                if (w_len > 4) zmm18 = _mm512_add_ps(zmm31, zmm18);
                if (w_len > 5) zmm19 = _mm512_add_ps(zmm31, zmm19);
                if (w_len > 6) zmm20 = _mm512_add_ps(zmm31, zmm20);
                if (w_len > 7) zmm21 = _mm512_add_ps(zmm31, zmm21);
                if (w_len > 8) zmm22 = _mm512_add_ps(zmm31, zmm22);
                if (w_len > 9) zmm23 = _mm512_add_ps(zmm31, zmm23);
                if (w_len > 10) zmm24 = _mm512_add_ps(zmm31, zmm24);
                if (w_len > 11) zmm25 = _mm512_add_ps(zmm31, zmm25);
                if (w_len > 12) zmm26 = _mm512_add_ps(zmm31, zmm26);
                if (w_len > 13) zmm27 = _mm512_add_ps(zmm31, zmm27);
            }
        }
        
        const float *icb_src = src + kh_start * (src_dh_stride + kernel_w * src_dw_stride);
        const float *icb_flt = PICK_PARAM(const float*, priv_param, FLT_IDX()) + kh_start * kernel_w * CH_DT_BLK() * CH_DT_BLK();
        int64_t channels     = shar_param[CHANNELS_IDX()];
        while (channels >= CH_DT_BLK()) {
            channels -= CH_DT_BLK();
            const float *k_src = icb_src;
            const float *k_flt = icb_flt;
            for (int64_t kh = kh_start; kh < kh_end; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    if (oc_len == 2 * CH_DT_BLK() && w_len == 14) {
                        const float *ic_src = k_src;
                        const float *ic_flt = k_flt;
                        IC_COMPUTE_STEP(0);
                        IC_COMPUTE_STEP(1);
                        IC_COMPUTE_STEP(2);
                        IC_COMPUTE_STEP(3);

                        IC_COMPUTE_STEP(4);
                        IC_COMPUTE_STEP(5);
                        IC_COMPUTE_STEP(6);
                        IC_COMPUTE_STEP(7);

                        IC_COMPUTE_STEP(8);
                        IC_COMPUTE_STEP(9);
                        IC_COMPUTE_STEP(10);
                        IC_COMPUTE_STEP(11);

                        IC_COMPUTE_STEP(12);
                        IC_COMPUTE_STEP(13);
                        IC_COMPUTE_STEP(14);
                        IC_COMPUTE_STEP(15);
                    } else {
                        const float *ic_src = k_src;
                        const float *ic_flt = k_flt;
                        for (int64_t ic = 0; ic < CH_DT_BLK(); ++ic) {
                            IC_COMPUTE_STEP(0);
                            ic_src += 1;
                            ic_flt += CH_DT_BLK();
                        }
                    }
                    k_flt += CH_DT_BLK() * CH_DT_BLK();
                    k_src += src_dw_stride;
                }
                k_src += src_dh_stride;
            }
            icb_flt += kernel_h * kernel_w * CH_DT_BLK() * CH_DT_BLK();
            icb_src += src_icb_stride;
        }
        if (channels > 0) {
            const float *k_src = icb_src;
            const float *k_flt = icb_flt;
            for (int64_t kh = kh_start; kh < kh_end; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    const float *ic_src = k_src;
                    const float *ic_flt = k_flt;
                    for (int64_t ic = 0; ic < channels; ++ic) {
                        IC_COMPUTE_STEP(0);
                        ic_src += 1;
                        ic_flt += CH_DT_BLK();
                    }
                    k_flt += CH_DT_BLK() * CH_DT_BLK();
                    k_src += src_dw_stride;
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
                if (w_len > 9) zmm9 = _mm512_max_ps(zmm9, zmm30);
                if (w_len > 10) zmm10 = _mm512_max_ps(zmm10, zmm30);
                if (w_len > 11) zmm11 = _mm512_max_ps(zmm11, zmm30);
                if (w_len > 12) zmm12 = _mm512_max_ps(zmm12, zmm30);
                if (w_len > 13) zmm13 = _mm512_max_ps(zmm13, zmm30);
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                if (w_len > 0) zmm14 = _mm512_max_ps(zmm14, zmm30);
                if (w_len > 1) zmm15 = _mm512_max_ps(zmm15, zmm30);
                if (w_len > 2) zmm16 = _mm512_max_ps(zmm16, zmm30);
                if (w_len > 3) zmm17 = _mm512_max_ps(zmm17, zmm30);
                if (w_len > 4) zmm18 = _mm512_max_ps(zmm18, zmm30);
                if (w_len > 5) zmm19 = _mm512_max_ps(zmm19, zmm30);
                if (w_len > 6) zmm20 = _mm512_max_ps(zmm20, zmm30);
                if (w_len > 7) zmm21 = _mm512_max_ps(zmm21, zmm30);
                if (w_len > 8) zmm22 = _mm512_max_ps(zmm22, zmm30);
                if (w_len > 9) zmm23 = _mm512_max_ps(zmm23, zmm30);
                if (w_len > 10) zmm24 = _mm512_max_ps(zmm24, zmm30);
                if (w_len > 11) zmm25 = _mm512_max_ps(zmm25, zmm30);
                if (w_len > 12) zmm26 = _mm512_max_ps(zmm26, zmm30);
                if (w_len > 13) zmm27 = _mm512_max_ps(zmm27, zmm30);
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
                if (w_len > 9) zmm9 = _mm512_min_ps(zmm9, zmm31);
                if (w_len > 10) zmm10 = _mm512_min_ps(zmm10, zmm31);
                if (w_len > 11) zmm11 = _mm512_min_ps(zmm11, zmm31);
                if (w_len > 12) zmm12 = _mm512_min_ps(zmm12, zmm31);
                if (w_len > 13) zmm13 = _mm512_min_ps(zmm13, zmm31);
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                if (w_len > 0) zmm14 = _mm512_min_ps(zmm14, zmm31);
                if (w_len > 1) zmm15 = _mm512_min_ps(zmm15, zmm31);
                if (w_len > 2) zmm16 = _mm512_min_ps(zmm16, zmm31);
                if (w_len > 3) zmm17 = _mm512_min_ps(zmm17, zmm31);
                if (w_len > 4) zmm18 = _mm512_min_ps(zmm18, zmm31);
                if (w_len > 5) zmm19 = _mm512_min_ps(zmm19, zmm31);
                if (w_len > 6) zmm20 = _mm512_min_ps(zmm20, zmm31);
                if (w_len > 7) zmm21 = _mm512_min_ps(zmm21, zmm31);
                if (w_len > 8) zmm22 = _mm512_min_ps(zmm22, zmm31);
                if (w_len > 9) zmm23 = _mm512_min_ps(zmm23, zmm31);
                if (w_len > 10) zmm24 = _mm512_min_ps(zmm24, zmm31);
                if (w_len > 11) zmm25 = _mm512_min_ps(zmm25, zmm31);
                if (w_len > 12) zmm26 = _mm512_min_ps(zmm26, zmm31);
                if (w_len > 13) zmm27 = _mm512_min_ps(zmm27, zmm31);
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
                if (w_len > 9) _mm512_stream_ps(l_dst + 9 * CH_DT_BLK(), zmm9);
                if (w_len > 10) _mm512_stream_ps(l_dst + 10 * CH_DT_BLK(), zmm10);
                if (w_len > 11) _mm512_stream_ps(l_dst + 11 * CH_DT_BLK(), zmm11);
                if (w_len > 12) _mm512_stream_ps(l_dst + 12 * CH_DT_BLK(), zmm12);
                if (w_len > 13) _mm512_stream_ps(l_dst + 13 * CH_DT_BLK(), zmm13);
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                if (w_len > 0) _mm512_stream_ps(l_dst + 0 * CH_DT_BLK(), zmm14);
                if (w_len > 1) _mm512_stream_ps(l_dst + 1 * CH_DT_BLK(), zmm15);
                if (w_len > 2) _mm512_stream_ps(l_dst + 2 * CH_DT_BLK(), zmm16);
                if (w_len > 3) _mm512_stream_ps(l_dst + 3 * CH_DT_BLK(), zmm17);
                if (w_len > 4) _mm512_stream_ps(l_dst + 4 * CH_DT_BLK(), zmm18);
                if (w_len > 5) _mm512_stream_ps(l_dst + 5 * CH_DT_BLK(), zmm19);
                if (w_len > 6) _mm512_stream_ps(l_dst + 6 * CH_DT_BLK(), zmm20);
                if (w_len > 7) _mm512_stream_ps(l_dst + 7 * CH_DT_BLK(), zmm21);
                if (w_len > 8) _mm512_stream_ps(l_dst + 8 * CH_DT_BLK(), zmm22);
                if (w_len > 9) _mm512_stream_ps(l_dst + 9 * CH_DT_BLK(), zmm23);
                if (w_len > 10) _mm512_stream_ps(l_dst + 10 * CH_DT_BLK(), zmm24);
                if (w_len > 11) _mm512_stream_ps(l_dst + 11 * CH_DT_BLK(), zmm25);
                if (w_len > 12) _mm512_stream_ps(l_dst + 12 * CH_DT_BLK(), zmm26);
                if (w_len > 13) _mm512_stream_ps(l_dst + 13 * CH_DT_BLK(), zmm27);
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
                if (w_len > 9) _mm512_storeu_ps(l_dst + 9 * CH_DT_BLK(), zmm9);
                if (w_len > 10) _mm512_storeu_ps(l_dst + 10 * CH_DT_BLK(), zmm10);
                if (w_len > 11) _mm512_storeu_ps(l_dst + 11 * CH_DT_BLK(), zmm11);
                if (w_len > 12) _mm512_storeu_ps(l_dst + 12 * CH_DT_BLK(), zmm12);
                if (w_len > 13) _mm512_storeu_ps(l_dst + 13 * CH_DT_BLK(), zmm13);
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                if (w_len > 0) _mm512_storeu_ps(l_dst + 0 * CH_DT_BLK(), zmm14);
                if (w_len > 1) _mm512_storeu_ps(l_dst + 1 * CH_DT_BLK(), zmm15);
                if (w_len > 2) _mm512_storeu_ps(l_dst + 2 * CH_DT_BLK(), zmm16);
                if (w_len > 3) _mm512_storeu_ps(l_dst + 3 * CH_DT_BLK(), zmm17);
                if (w_len > 4) _mm512_storeu_ps(l_dst + 4 * CH_DT_BLK(), zmm18);
                if (w_len > 5) _mm512_storeu_ps(l_dst + 5 * CH_DT_BLK(), zmm19);
                if (w_len > 6) _mm512_storeu_ps(l_dst + 6 * CH_DT_BLK(), zmm20);
                if (w_len > 7) _mm512_storeu_ps(l_dst + 7 * CH_DT_BLK(), zmm21);
                if (w_len > 8) _mm512_storeu_ps(l_dst + 8 * CH_DT_BLK(), zmm22);
                if (w_len > 9) _mm512_storeu_ps(l_dst + 9 * CH_DT_BLK(), zmm23);
                if (w_len > 10) _mm512_storeu_ps(l_dst + 10 * CH_DT_BLK(), zmm24);
                if (w_len > 11) _mm512_storeu_ps(l_dst + 11 * CH_DT_BLK(), zmm25);
                if (w_len > 12) _mm512_storeu_ps(l_dst + 12 * CH_DT_BLK(), zmm26);
                if (w_len > 13) _mm512_storeu_ps(l_dst + 13 * CH_DT_BLK(), zmm27);
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
