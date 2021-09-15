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
#include <nmmintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/depthwise/sse/conv2d_depthwise_kernel_fp32_sse.h"

namespace ppl { namespace kernel { namespace x86 {

template <int32_t spec_stride_w, int32_t w_len>
void conv2d_depthwise_fp32_sse_kernel(
    const int64_t *shar_param,
    int64_t *priv_param)
{
#define KW_COMPUTE_STEP() do {\
    xmm14 = _mm_loadu_ps(k_flt);\
    if (w_len > 0) {\
        xmm15 = _mm_loadu_ps(k_src_0 + 0 * src_sw_stride);\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm0 = _mm_add_ps(xmm0, xmm15);\
    }\
    if (w_len > 1) {\
        xmm15 = _mm_loadu_ps(k_src_0 + 1 * src_sw_stride);\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm1 = _mm_add_ps(xmm1, xmm15);\
    }\
    if (w_len > 2) {\
        xmm15 = _mm_loadu_ps(k_src_0 + 2 * src_sw_stride);\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm2 = _mm_add_ps(xmm2, xmm15);\
    }\
    if (w_len > 0) k_src_0 += src_dw_stride;\
    if (w_len > 3) {\
        xmm15 = _mm_loadu_ps(k_src_3 + 0 * src_sw_stride);\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm3 = _mm_add_ps(xmm3, xmm15);\
    }\
    if (w_len > 4) {\
        xmm15 = _mm_loadu_ps(k_src_3 + 1 * src_sw_stride);\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm4 = _mm_add_ps(xmm4, xmm15);\
    }\
    if (w_len > 5) {\
        xmm15 = _mm_loadu_ps(k_src_3 + 2 * src_sw_stride);\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm5 = _mm_add_ps(xmm5, xmm15);\
    }\
    if (w_len > 3) k_src_3 += src_dw_stride;\
    if (w_len > 6) {\
        xmm15 = _mm_loadu_ps(k_src_6 + 0 * src_sw_stride);\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm6 = _mm_add_ps(xmm6, xmm15);\
    }\
    if (w_len > 7) {\
        xmm15 = _mm_loadu_ps(k_src_6 + 1 * src_sw_stride);\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm7 = _mm_add_ps(xmm7, xmm15);\
    }\
    if (w_len > 8) {\
        xmm15 = _mm_loadu_ps(k_src_6 + 2 * src_sw_stride);\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm8 = _mm_add_ps(xmm8, xmm15);\
    }\
    if (w_len > 6) k_src_6 += src_dw_stride;\
    if (w_len > 9) {\
        xmm15 = _mm_loadu_ps(k_src_9 + 0 * src_sw_stride);\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm9 = _mm_add_ps(xmm9, xmm15);\
    }\
    if (w_len > 10) {\
        xmm15 = _mm_loadu_ps(k_src_9 + 1 * src_sw_stride);\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm10 = _mm_add_ps(xmm10, xmm15);\
    }\
    if (w_len > 11) {\
        xmm15 = _mm_loadu_ps(k_src_9 + 2 * src_sw_stride);\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm11 = _mm_add_ps(xmm11, xmm15);\
    }\
    if (w_len > 9) k_src_9 += src_dw_stride;\
    if (w_len > 12) {\
        xmm15 = _mm_loadu_ps(k_src_12 + 0 * src_sw_stride);\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm12 = _mm_add_ps(xmm12, xmm15);\
    }\
    if (w_len > 13) {\
        xmm15 = _mm_loadu_ps(k_src_12 + 1 * src_sw_stride);\
        xmm15 = _mm_mul_ps(xmm15, xmm14);\
        xmm13 = _mm_add_ps(xmm13, xmm15);\
    }\
    if (w_len > 12) k_src_12 += src_dw_stride;\
    k_flt += CH_DT_BLK();\
} while (0)

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
    __m128 xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15;

    const int64_t src_sw_stride = spec_stride_w ? spec_stride_w * CH_DT_BLK() : shar_param[SRC_SW_STRIDE_IDX()];
    const int64_t src_dh_stride = shar_param[SRC_DH_STRIDE_IDX()];
    const int64_t src_dw_stride = shar_param[SRC_DW_STRIDE_IDX()];
    const int64_t kernel_w      = shar_param[KW_IDX()];
    const int64_t src_kh_stride = src_dh_stride - kernel_w * src_dw_stride;

    const int64_t kh_start = priv_param[KH_START_IDX()];
    const int64_t kh_end   = priv_param[KH_END_IDX()];

    const float *src = PICK_PARAM(const float*, priv_param, SRC_IDX());
    float *dst       = PICK_PARAM(float*, priv_param, DST_IDX());
    int64_t ow       = priv_param[OW_IDX()];
    do {
        const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
        if (w_len > 0) xmm0 = _mm_loadu_ps(bias);
        if (w_len > 1) xmm1 = xmm0;
        if (w_len > 2) xmm2 = xmm0;
        if (w_len > 3) xmm3 = xmm0;
        if (w_len > 4) xmm4 = xmm0;
        if (w_len > 5) xmm5 = xmm0;
        if (w_len > 6) xmm6 = xmm0;
        if (w_len > 7) xmm7 = xmm0;
        if (w_len > 8) xmm8 = xmm0;
        if (w_len > 9) xmm9 = xmm0;
        if (w_len > 10) xmm10 = xmm0;
        if (w_len > 11) xmm11 = xmm0;
        if (w_len > 12) xmm12 = xmm0;
        if (w_len > 13) xmm13 = xmm0;

        const float *k_src_0 = w_len > 0 ? src + kh_start * src_dh_stride : nullptr;
        const float *k_src_3 = w_len > 3 ? k_src_0 + 3 * src_sw_stride : nullptr;
        const float *k_src_6 = w_len > 6 ? k_src_0 + 6 * src_sw_stride : nullptr;
        const float *k_src_9 = w_len > 9 ? k_src_0 + 9 * src_sw_stride : nullptr;
        const float *k_src_12 = w_len > 12 ? k_src_0 + 12 * src_sw_stride : nullptr;
        const float *k_flt  = PICK_PARAM(const float*, priv_param, FLT_IDX()) + kh_start * kernel_w * CH_DT_BLK();
        for (int32_t kh = kh_start; kh < kh_end; ++kh) {
            for (int32_t kw = 0; kw < kernel_w; ++kw) {
                KW_COMPUTE_STEP();
            }
            if(w_len > 0) k_src_0 += src_kh_stride;
            if(w_len > 3) k_src_3 += src_kh_stride;
            if(w_len > 6) k_src_6 += src_kh_stride;
            if(w_len > 9) k_src_9 += src_kh_stride;
            if(w_len > 12) k_src_12 += src_kh_stride;
        }
    
        if (w_len > 0) _mm_storeu_ps(dst + 0 * CH_DT_BLK(), xmm0);
        if (w_len > 1) _mm_storeu_ps(dst + 1 * CH_DT_BLK(), xmm1);
        if (w_len > 2) _mm_storeu_ps(dst + 2 * CH_DT_BLK(), xmm2);
        if (w_len > 3) _mm_storeu_ps(dst + 3 * CH_DT_BLK(), xmm3);
        if (w_len > 4) _mm_storeu_ps(dst + 4 * CH_DT_BLK(), xmm4);
        if (w_len > 5) _mm_storeu_ps(dst + 5 * CH_DT_BLK(), xmm5);
        if (w_len > 6) _mm_storeu_ps(dst + 6 * CH_DT_BLK(), xmm6);
        if (w_len > 7) _mm_storeu_ps(dst + 7 * CH_DT_BLK(), xmm7);
        if (w_len > 8) _mm_storeu_ps(dst + 8 * CH_DT_BLK(), xmm8);
        if (w_len > 9) _mm_storeu_ps(dst + 9 * CH_DT_BLK(), xmm9);
        if (w_len > 10) _mm_storeu_ps(dst + 10 * CH_DT_BLK(), xmm10);
        if (w_len > 11) _mm_storeu_ps(dst + 11 * CH_DT_BLK(), xmm11);
        if (w_len > 12) _mm_storeu_ps(dst + 12 * CH_DT_BLK(), xmm12);
        if (w_len > 13) _mm_storeu_ps(dst + 13 * CH_DT_BLK(), xmm13);
        
        src += w_len * src_sw_stride;
        dst += w_len * CH_DT_BLK();
        ow -= w_len;
    } while (ow > 0);
    PICK_PARAM(const float *, priv_param, SRC_IDX()) = src;
    PICK_PARAM(float *, priv_param, DST_IDX()) = dst;
#undef KW_COMPUTE_STEP
}

#define DEPTHWISE_KERNEL_TABLE_BLK(STRIDE_W) \
    {\
        conv2d_depthwise_fp32_sse_kernel<STRIDE_W, 1>,\
        conv2d_depthwise_fp32_sse_kernel<STRIDE_W, 2>,\
        conv2d_depthwise_fp32_sse_kernel<STRIDE_W, 3>,\
        conv2d_depthwise_fp32_sse_kernel<STRIDE_W, 4>,\
        conv2d_depthwise_fp32_sse_kernel<STRIDE_W, 5>,\
        conv2d_depthwise_fp32_sse_kernel<STRIDE_W, 6>,\
        conv2d_depthwise_fp32_sse_kernel<STRIDE_W, 7>,\
        conv2d_depthwise_fp32_sse_kernel<STRIDE_W, 8>,\
        conv2d_depthwise_fp32_sse_kernel<STRIDE_W, 9>,\
        conv2d_depthwise_fp32_sse_kernel<STRIDE_W, 10>,\
        conv2d_depthwise_fp32_sse_kernel<STRIDE_W, 11>,\
        conv2d_depthwise_fp32_sse_kernel<STRIDE_W, 12>,\
        conv2d_depthwise_fp32_sse_kernel<STRIDE_W, 13>,\
        conv2d_depthwise_fp32_sse_kernel<STRIDE_W, 14>,\
    }

conv2d_depthwise_kernel_fp32_sse_func_t
conv2d_depthwise_kernel_fp32_sse_table[STRIDE_W_OPT()][MAX_OW_RF()] =
{
    DEPTHWISE_KERNEL_TABLE_BLK(0),
    DEPTHWISE_KERNEL_TABLE_BLK(1),
    DEPTHWISE_KERNEL_TABLE_BLK(2),
};

}}};
