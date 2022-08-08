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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_NDARRAY_SSE_CONV2D_N8CX_DIRECT_NDARRAY_BLK1X1_KERNEL_FP32_SSE_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_NDARRAY_SSE_CONV2D_N8CX_DIRECT_NDARRAY_BLK1X1_KERNEL_FP32_SSE_H_

#include <nmmintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/sse/conv2d_n8cx_direct_ndarray_kernel_fp32_sse.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store, int32_t oc_len>
void conv2d_n8cx_direct_ndarray_fp32_sse_blk1x1_kernel(
    const int64_t *shar_param,
    int64_t *priv_param)
{
    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

    const int64_t kernel_h = shar_param[KH_IDX()];
    const int64_t kernel_w = shar_param[KW_IDX()];
    const int64_t stride_w = shar_param[SW_IDX()];
    const int64_t src_c_stride = shar_param[SRC_C_STRIDE_IDX()];
    const int64_t src_h_stride = shar_param[SRC_H_STRIDE_IDX()];
    const int64_t flt_ocb_stride = shar_param[FLT_OCB_STRIDE_IDX()];
    const int64_t kernel_flags = shar_param[FLAGS_IDX()];
    const int64_t kh_start = priv_param[KH_START_IDX()];
    const int64_t kh_end = priv_param[KH_END_IDX()];
    const int64_t kw_start = priv_param[KW_START_IDX()];
    const int64_t kw_end = priv_param[KW_END_IDX()];

    if (kernel_flags & KERNEL_FLAG_RELU6()) {
        xmm7 = _mm_set1_ps(6.0f);
    }

    if (kernel_flags & KERNEL_FLAG_LD_BIAS()) {
        const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
        if (oc_len > 0 * OC_DT_BLK()) {
            xmm0 = _mm_loadu_ps(bias + 0 * OC_DT_BLK() + 0 * OC_RF_BLK());
            xmm1 = _mm_loadu_ps(bias + 0 * OC_DT_BLK() + 1 * OC_RF_BLK());
        }
        if (oc_len > 1 * OC_DT_BLK()) {
            xmm2 = _mm_loadu_ps(bias + 1 * OC_DT_BLK() + 0 * OC_RF_BLK());
            xmm3 = _mm_loadu_ps(bias + 1 * OC_DT_BLK() + 1 * OC_RF_BLK());
        }
    } else {
        const float* his = PICK_PARAM(const float*, priv_param, HIS_IDX());
        const int64_t his_ocb_stride = shar_param[HIS_OCB_STRIDE_IDX()];
        if (oc_len > 0 * OC_DT_BLK()) {
            xmm0 = _mm_loadu_ps(his + 0 * OC_RF_BLK());
            xmm1 = _mm_loadu_ps(his + 1 * OC_RF_BLK());
        }
        if (oc_len > 1 * OC_DT_BLK()) {
            his += his_ocb_stride;
            xmm2 = _mm_loadu_ps(his + 0 * OC_RF_BLK());
            xmm3 = _mm_loadu_ps(his + 1 * OC_RF_BLK());
        }
    }

    if (kernel_flags & KERNEL_FLAG_AD_BIAS()) {
        const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
        if (oc_len > 0 * OC_DT_BLK()) {
            xmm0 = _mm_add_ps(_mm_loadu_ps(bias + 0 * OC_DT_BLK() + 0 * OC_RF_BLK()), xmm0);
            xmm1 = _mm_add_ps(_mm_loadu_ps(bias + 0 * OC_DT_BLK() + 1 * OC_RF_BLK()), xmm1);
        }
        if (oc_len > 1 * OC_DT_BLK()) {
            xmm2 = _mm_add_ps(_mm_loadu_ps(bias + 1 * OC_DT_BLK() + 0 * OC_RF_BLK()), xmm2);
            xmm3 = _mm_add_ps(_mm_loadu_ps(bias + 1 * OC_DT_BLK() + 1 * OC_RF_BLK()), xmm3);
        }
    }

    const float *ic_src = PICK_PARAM(const float*, priv_param, SRC_IDX()) + kh_start * src_h_stride;
    const float *ic_flt = PICK_PARAM(const float*, priv_param, FLT_IDX()) + kh_start * kernel_w * OC_DT_BLK();
    int64_t channels = shar_param[CHANNELS_IDX()];
    do {
        const float *k_src = ic_src;
        const float *k_flt = ic_flt;
        for (int64_t kh = kh_start; kh < kh_end; ++kh) {
            for (int64_t kw = kw_start; kw < kw_end; ++kw) {
                xmm4 = _mm_set1_ps(k_src[kw]);
                if (oc_len > 0 * OC_DT_BLK()) {
                    xmm5 = _mm_loadu_ps(k_flt + 0 * flt_ocb_stride + kw * OC_DT_BLK() + 0 * OC_RF_BLK());
                    xmm6 = _mm_loadu_ps(k_flt + 0 * flt_ocb_stride + kw * OC_DT_BLK() + 1 * OC_RF_BLK());
                    xmm5 = _mm_mul_ps(xmm5, xmm4);
                    xmm6 = _mm_mul_ps(xmm6, xmm4);
                    xmm0 = _mm_add_ps(xmm0, xmm5);
                    xmm1 = _mm_add_ps(xmm1, xmm6);
                }
                if (oc_len > 1 * OC_DT_BLK()) {
                    xmm5 = _mm_loadu_ps(k_flt + 1 * flt_ocb_stride + kw * OC_DT_BLK() + 0 * OC_RF_BLK());
                    xmm6 = _mm_loadu_ps(k_flt + 1 * flt_ocb_stride + kw * OC_DT_BLK() + 1 * OC_RF_BLK());
                    xmm5 = _mm_mul_ps(xmm5, xmm4);
                    xmm6 = _mm_mul_ps(xmm6, xmm4);
                    xmm2 = _mm_add_ps(xmm2, xmm5);
                    xmm3 = _mm_add_ps(xmm3, xmm6);
                }
            }
            k_flt += kernel_w * OC_DT_BLK();
            k_src += src_h_stride;
        }
        ic_flt += kernel_h * kernel_w * OC_DT_BLK();
        ic_src += src_c_stride;
        channels -= 1;
    } while (channels > 0);
    
    if (kernel_flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
        xmm4 = _mm_setzero_ps();
        if (oc_len > 0 * OC_DT_BLK()) {
            xmm0 = _mm_max_ps(xmm0, xmm4);
            xmm1 = _mm_max_ps(xmm1, xmm4);
        }
        if (oc_len > 1 * OC_DT_BLK()) {
            xmm2 = _mm_max_ps(xmm2, xmm4);
            xmm3 = _mm_max_ps(xmm3, xmm4);
        }
    }
    if (kernel_flags & KERNEL_FLAG_RELU6()) {
        if (oc_len > 0 * OC_DT_BLK()) {
            xmm0 = _mm_min_ps(xmm0, xmm7);
            xmm1 = _mm_min_ps(xmm1, xmm7);
        }
        if (oc_len > 1 * OC_DT_BLK()) {
            xmm2 = _mm_min_ps(xmm2, xmm7);
            xmm3 = _mm_min_ps(xmm3, xmm7);
        }
    }

    if (nt_store) {
        float* dst = PICK_PARAM(float*, priv_param, DST_IDX());
        const int64_t dst_ocb_stride = shar_param[DST_OCB_STRIDE_IDX()];
        if (oc_len > 0 * OC_DT_BLK()) {
            _mm_stream_ps(dst + 0 * OC_RF_BLK(), xmm0);
            _mm_stream_ps(dst + 1 * OC_RF_BLK(), xmm1);
        }
        if (oc_len > 1 * OC_DT_BLK()) {
            dst += dst_ocb_stride;
            _mm_stream_ps(dst + 0 * OC_RF_BLK(), xmm2);
            _mm_stream_ps(dst + 1 * OC_RF_BLK(), xmm3);
        }
    } else {
        float* dst = PICK_PARAM(float*, priv_param, DST_IDX());
        const int64_t dst_ocb_stride = shar_param[DST_OCB_STRIDE_IDX()];
        if (oc_len > 0 * OC_DT_BLK()) {
            _mm_storeu_ps(dst + 0 * OC_RF_BLK(), xmm0);
            _mm_storeu_ps(dst + 1 * OC_RF_BLK(), xmm1);
        }
        if (oc_len > 1 * OC_DT_BLK()) {
            dst += dst_ocb_stride;
            _mm_storeu_ps(dst + 0 * OC_RF_BLK(), xmm2);
            _mm_storeu_ps(dst + 1 * OC_RF_BLK(), xmm3);
        }
    }
    PICK_PARAM(const float*, priv_param, SRC_IDX()) += stride_w;
    PICK_PARAM(const float*, priv_param, HIS_IDX()) += OC_DT_BLK();
    PICK_PARAM(float*, priv_param, DST_IDX()) += OC_DT_BLK();
}

}}};

#endif
