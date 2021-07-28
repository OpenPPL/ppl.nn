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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_SSE_CONV2D_N8CX_DIRECT_BLK1X3_KERNEL_FP32_SSE_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_SSE_CONV2D_N8CX_DIRECT_BLK1X3_KERNEL_FP32_SSE_H_

#include <nmmintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/direct/sse/conv2d_n8cx_direct_kernel_fp32_sse.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store, int32_t spec_stride_w, int32_t oc_len, int32_t w_len>
void conv2d_n8cx_direct_fp32_sse_blk1x3_kernel(
    const int64_t *shar_param,
    int64_t *priv_param)
{
#define IC_COMPUTE_STEP(IC) do {\
    if (w_len > 0) {\
        xmm12 = _mm_set1_ps(ic_src[(IC) + 0 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) {\
            xmm15 = _mm_loadu_ps(ic_flt_o8 + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm12);\
            xmm0 = _mm_add_ps(xmm0, xmm15);\
            xmm15 = _mm_loadu_ps(ic_flt_o8 + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm12);\
            xmm1 = _mm_add_ps(xmm1, xmm15);\
        }\
        if (oc_len > 1 * CH_DT_BLK()) {\
            xmm15 = _mm_loadu_ps(ic_flt_o16 + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm12);\
            xmm6 = _mm_add_ps(xmm6, xmm15);\
            xmm15 = _mm_loadu_ps(ic_flt_o16 + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm12);\
            xmm7 = _mm_add_ps(xmm7, xmm15);\
        }\
    }\
    if (w_len > 1) {\
        xmm13 = _mm_set1_ps(ic_src[(IC) + 1 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) {\
            xmm15 = _mm_loadu_ps(ic_flt_o8 + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm13);\
            xmm2 = _mm_add_ps(xmm2, xmm15);\
            xmm15 = _mm_loadu_ps(ic_flt_o8 + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm13);\
            xmm3 = _mm_add_ps(xmm3, xmm15);\
        }\
        if (oc_len > 1 * CH_DT_BLK()) {\
            xmm15 = _mm_loadu_ps(ic_flt_o16 + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm13);\
            xmm8 = _mm_add_ps(xmm8, xmm15);\
            xmm15 = _mm_loadu_ps(ic_flt_o16 + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm13);\
            xmm9 = _mm_add_ps(xmm9, xmm15);\
        }\
    }\
    if (w_len > 2) {\
        xmm14 = _mm_set1_ps(ic_src[(IC) + 2 * src_sw_stride]);\
        if (oc_len > 0 * CH_DT_BLK()) {\
            xmm15 = _mm_loadu_ps(ic_flt_o8 + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm14);\
            xmm4 = _mm_add_ps(xmm4, xmm15);\
            xmm15 = _mm_loadu_ps(ic_flt_o8 + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm14);\
            xmm5 = _mm_add_ps(xmm5, xmm15);\
        }\
        if (oc_len > 1 * CH_DT_BLK()) {\
            xmm15 = _mm_loadu_ps(ic_flt_o16 + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm14);\
            xmm10 = _mm_add_ps(xmm10, xmm15);\
            xmm15 = _mm_loadu_ps(ic_flt_o16 + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm14);\
            xmm11 = _mm_add_ps(xmm11, xmm15);\
        }\
    }\
} while (0)

#define IC_PREFETCH_STEP(IC)  do {\
    if (oc_len > 1 * CH_DT_BLK() && w_len > 1) {\
        if (oc_len > 0 * CH_DT_BLK()) _mm_prefetch((const char*)ic_flt_o8 + 0 * flt_ocb_stride + (IC) * CH_DT_BLK() + CH_DT_BLK() * CH_DT_BLK(), _MM_HINT_T0);\
        if (oc_len > 1 * CH_DT_BLK()) _mm_prefetch((const char*)ic_flt_o16 + 0 * flt_ocb_stride + (IC) * CH_DT_BLK() + CH_DT_BLK() * CH_DT_BLK(), _MM_HINT_T0);\
    }\
} while (0)

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
    __m128 xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15;

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
                if (w_len > 0) {
                    xmm0 = _mm_loadu_ps(bias + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
                    xmm1 = _mm_loadu_ps(bias + 0 * CH_DT_BLK() + 1 * CH_RF_BLK());
                }
                if (w_len > 1) {
                    xmm2 = xmm0;
                    xmm3 = xmm1;
                }
                if (w_len > 2) {
                    xmm4 = xmm0;
                    xmm5 = xmm1;
                }
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                if (w_len > 0) {
                    xmm6 = _mm_loadu_ps(bias + 1 * CH_DT_BLK() + 0 * CH_RF_BLK());
                    xmm7 = _mm_loadu_ps(bias + 1 * CH_DT_BLK() + 1 * CH_RF_BLK());
                }
                if (w_len > 1) {
                    xmm8 = xmm6;
                    xmm9 = xmm7;
                }
                if (w_len > 2) {
                    xmm10 = xmm6;
                    xmm11 = xmm7;
                }
            }
        } else {
            const float *l_his = his;
            const int64_t his_ocb_stride = shar_param[HIS_OCB_STRIDE_IDX()];
            if (oc_len > 0 * CH_DT_BLK()) {
                if (w_len > 0) {
                    xmm0 = _mm_loadu_ps(l_his + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
                    xmm1 = _mm_loadu_ps(l_his + 0 * CH_DT_BLK() + 1 * CH_RF_BLK());
                }
                if (w_len > 1) {
                    xmm2 = _mm_loadu_ps(l_his + 1 * CH_DT_BLK() + 0 * CH_RF_BLK());
                    xmm3 = _mm_loadu_ps(l_his + 1 * CH_DT_BLK() + 1 * CH_RF_BLK());
                }
                if (w_len > 2) {
                    xmm4 = _mm_loadu_ps(l_his + 2 * CH_DT_BLK() + 0 * CH_RF_BLK());
                    xmm5 = _mm_loadu_ps(l_his + 2 * CH_DT_BLK() + 1 * CH_RF_BLK());
                }
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                l_his += his_ocb_stride;
                if (w_len > 0) {
                    xmm6 = _mm_loadu_ps(l_his + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
                    xmm7 = _mm_loadu_ps(l_his + 0 * CH_DT_BLK() + 1 * CH_RF_BLK());
                }
                if (w_len > 1) {
                    xmm8 = _mm_loadu_ps(l_his + 1 * CH_DT_BLK() + 0 * CH_RF_BLK());
                    xmm9 = _mm_loadu_ps(l_his + 1 * CH_DT_BLK() + 1 * CH_RF_BLK());
                }
                if (w_len > 2) {
                    xmm10 = _mm_loadu_ps(l_his + 2 * CH_DT_BLK() + 0 * CH_RF_BLK());
                    xmm11 = _mm_loadu_ps(l_his + 2 * CH_DT_BLK() + 1 * CH_RF_BLK());
                }
            }
        }

        if (kernel_flags & KERNEL_FLAG_AD_BIAS()) {
            const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
            if (oc_len > 0 * CH_DT_BLK()) {
                xmm12 = _mm_loadu_ps(bias + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm13 = _mm_loadu_ps(bias + 0 * CH_DT_BLK() + 1 * CH_RF_BLK());
                if (w_len > 0) {
                    xmm0 = _mm_add_ps(xmm0, xmm12);
                    xmm1 = _mm_add_ps(xmm1, xmm13);
                }
                if (w_len > 1) {
                    xmm2 = _mm_add_ps(xmm2, xmm12);
                    xmm3 = _mm_add_ps(xmm3, xmm13);
                }
                if (w_len > 2) {
                    xmm4 = _mm_add_ps(xmm4, xmm12);
                    xmm5 = _mm_add_ps(xmm5, xmm13);
                }
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                xmm14 = _mm_loadu_ps(bias + 1 * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm15 = _mm_loadu_ps(bias + 1 * CH_DT_BLK() + 1 * CH_RF_BLK());
                if (w_len > 0) {
                    xmm6 = _mm_add_ps(xmm6, xmm14);
                    xmm7 = _mm_add_ps(xmm7, xmm15);
                }
                if (w_len > 1) {
                    xmm8 = _mm_add_ps(xmm8, xmm14);
                    xmm9 = _mm_add_ps(xmm9, xmm15);
                }
                if (w_len > 2) {
                    xmm10 = _mm_add_ps(xmm10, xmm14);
                    xmm11 = _mm_add_ps(xmm11, xmm15);
                }
            }
        }
        
        const float *icb_src = src + kh_start * (src_dh_stride + kernel_w * src_dw_stride);
        const float *icb_flt = PICK_PARAM(const float*, priv_param, FLT_IDX()) + kh_start * kernel_w * CH_DT_BLK() * CH_DT_BLK();
        int64_t channels     = shar_param[CHANNELS_IDX()];
        while (channels >= CH_DT_BLK()) {
            channels -= CH_DT_BLK();
            const float *k_src = icb_src;
            const float *k_flt_o8 = icb_flt + 0 * flt_ocb_stride;
            const float *k_flt_o16 = icb_flt + 1 * flt_ocb_stride;
            for (int64_t kh = kh_start; kh < kh_end; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    const float *ic_src = k_src;
                    const float *ic_flt_o8 = k_flt_o8;
                    const float *ic_flt_o16 = k_flt_o16;
                    for (int64_t ic = 0; ic < CH_DT_BLK(); ++ic) {
                        IC_COMPUTE_STEP(0);
                        IC_PREFETCH_STEP(0);
                        ic_src += 1;
                        ic_flt_o8 += CH_DT_BLK();
                        ic_flt_o16 += CH_DT_BLK();
                    }
                    k_flt_o8 += CH_DT_BLK() * CH_DT_BLK();
                    k_flt_o16 += CH_DT_BLK() * CH_DT_BLK();
                    k_src += src_dw_stride;
                }
                k_src += src_dh_stride;
            }
            icb_flt += kernel_h * kernel_w * CH_DT_BLK() * CH_DT_BLK();
            icb_src += src_icb_stride;
        }
        if (channels > 0) {
            const float *k_src = icb_src;
            const float *k_flt_o8 = icb_flt + 0 * flt_ocb_stride;
            const float *k_flt_o16 = icb_flt + 1 * flt_ocb_stride;
            for (int64_t kh = kh_start; kh < kh_end; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    const float *ic_src = k_src;
                    const float *ic_flt_o8 = k_flt_o8;
                    const float *ic_flt_o16 = k_flt_o16;
                    for (int64_t ic = 0; ic < channels; ++ic) {
                        IC_COMPUTE_STEP(0);
                        ic_src += 1;
                        ic_flt_o8 += CH_DT_BLK();
                        ic_flt_o16 += CH_DT_BLK();
                    }
                    k_flt_o8 += CH_DT_BLK() * CH_DT_BLK();
                    k_flt_o16 += CH_DT_BLK() * CH_DT_BLK();
                    k_src += src_dw_stride;
                }
                k_src += src_dh_stride;
            }
        }
        
        if (kernel_flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
            xmm12 = _mm_setzero_ps();
            if (oc_len > 0 * CH_DT_BLK()) {
                if (w_len > 0) {
                    xmm0 = _mm_max_ps(xmm0, xmm12);
                    xmm1 = _mm_max_ps(xmm1, xmm12);
                }
                if (w_len > 1) {
                    xmm2 = _mm_max_ps(xmm2, xmm12);
                    xmm3 = _mm_max_ps(xmm3, xmm12);
                }
                if (w_len > 2) {
                    xmm4 = _mm_max_ps(xmm4, xmm12);
                    xmm5 = _mm_max_ps(xmm5, xmm12);
                }
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                if (w_len > 0) {
                    xmm6 = _mm_max_ps(xmm6, xmm12);
                    xmm7 = _mm_max_ps(xmm7, xmm12);
                }
                if (w_len > 1) {
                    xmm8 = _mm_max_ps(xmm8, xmm12);
                    xmm9 = _mm_max_ps(xmm9, xmm12);
                }
                if (w_len > 2) {
                    xmm10 = _mm_max_ps(xmm10, xmm12);
                    xmm11 = _mm_max_ps(xmm11, xmm12);
                }
            }
        }
        if (kernel_flags & KERNEL_FLAG_RELU6()) {
            xmm13 = _mm_set1_ps(6.0f);
            if (oc_len > 0 * CH_DT_BLK()) {
                if (w_len > 0) {
                    xmm0 = _mm_min_ps(xmm0, xmm13);
                    xmm1 = _mm_min_ps(xmm1, xmm13);
                }
                if (w_len > 1) {
                    xmm2 = _mm_min_ps(xmm2, xmm13);
                    xmm3 = _mm_min_ps(xmm3, xmm13);
                }
                if (w_len > 2) {
                    xmm4 = _mm_min_ps(xmm4, xmm13);
                    xmm5 = _mm_min_ps(xmm5, xmm13);
                }
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                if (w_len > 0) {
                    xmm6 = _mm_min_ps(xmm6, xmm13);
                    xmm7 = _mm_min_ps(xmm7, xmm13);
                }
                if (w_len > 1) {
                    xmm8 = _mm_min_ps(xmm8, xmm13);
                    xmm9 = _mm_min_ps(xmm9, xmm13);
                }
                if (w_len > 2) {
                    xmm10 = _mm_min_ps(xmm10, xmm13);
                    xmm11 = _mm_min_ps(xmm11, xmm13);
                }
            }
        }

        if (nt_store) {
            float* l_dst = dst;
            const int64_t dst_ocb_stride = shar_param[DST_OCB_STRIDE_IDX()];
            if (oc_len > 0 * CH_DT_BLK()) {
                if (w_len > 0) {
                    _mm_stream_ps(l_dst + 0 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm0);
                    _mm_stream_ps(l_dst + 0 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm1);
                }
                if (w_len > 1) {
                    _mm_stream_ps(l_dst + 1 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm2);
                    _mm_stream_ps(l_dst + 1 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm3);
                }
                if (w_len > 2) {
                    _mm_stream_ps(l_dst + 2 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm4);
                    _mm_stream_ps(l_dst + 2 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm5);
                }
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                if (w_len > 0) {
                    _mm_stream_ps(l_dst + 0 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm6);
                    _mm_stream_ps(l_dst + 0 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm7);
                }
                if (w_len > 1) {
                    _mm_stream_ps(l_dst + 1 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm8);
                    _mm_stream_ps(l_dst + 1 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm9);
                }
                if (w_len > 2) {
                    _mm_stream_ps(l_dst + 2 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm10);
                    _mm_stream_ps(l_dst + 2 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm11);
                }
            }
        } else {
            float* l_dst = dst;
            const int64_t dst_ocb_stride = shar_param[DST_OCB_STRIDE_IDX()];
            if (oc_len > 0 * CH_DT_BLK()) {
                if (w_len > 0) {
                    _mm_storeu_ps(l_dst + 0 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm0);
                    _mm_storeu_ps(l_dst + 0 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm1);
                }
                if (w_len > 1) {
                    _mm_storeu_ps(l_dst + 1 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm2);
                    _mm_storeu_ps(l_dst + 1 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm3);
                }
                if (w_len > 2) {
                    _mm_storeu_ps(l_dst + 2 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm4);
                    _mm_storeu_ps(l_dst + 2 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm5);
                }
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                if (w_len > 0) {
                    _mm_storeu_ps(l_dst + 0 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm6);
                    _mm_storeu_ps(l_dst + 0 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm7);
                }
                if (w_len > 1) {
                    _mm_storeu_ps(l_dst + 1 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm8);
                    _mm_storeu_ps(l_dst + 1 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm9);
                }
                if (w_len > 2) {
                    _mm_storeu_ps(l_dst + 2 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm10);
                    _mm_storeu_ps(l_dst + 2 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm11);
                }
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
