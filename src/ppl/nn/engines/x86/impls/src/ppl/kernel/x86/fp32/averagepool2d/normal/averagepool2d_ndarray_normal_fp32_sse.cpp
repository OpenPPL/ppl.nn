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
#include <string.h>
#include "ppl/kernel/x86/common/sse_tools.h"
#include "ppl/kernel/x86/fp32/transpose/sse/transpose_fp32_sse.h"
#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/nn/params/onnx/pooling_param.h"
#include "ppl/kernel/x86/common/averagepool2d/averagepool2d_common.h"
#include "ppl/kernel/x86/fp32/reduce/sse/reduce_ndarray_fp32_sse.h"

#define C_BLK() ((int64_t)4)

namespace ppl { namespace kernel { namespace x86 {

uint64_t averagepool_fp32_get_buffer_bytes(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t pad_w)
{
    const int64_t src_h         = src_shape->GetDim(2);
    const int64_t src_w         = src_shape->GetDim(3);
    const int64_t padded_src_w  = src_w + pad_w;
    const int64_t dst_w         = dst_shape->GetDim(3);
    const int64_t src_trans_len = src_h * padded_src_w * C_BLK();
    const int64_t dst_trans_len = dst_w * C_BLK();
    return PPL_OMP_MAX_THREADS() * sizeof(float) * ((uint64_t)(src_trans_len + dst_trans_len));
}

static void pooling2d_fp32_sse_dst_trans(
    const float *dst_trans,
    int64_t width,
    int64_t channels,
    int64_t dst_c_stride,
    float *dst)
{
    const float *l_dst_trans = dst_trans;
    float *l_dst             = dst;
    const int64_t unroll_w   = 4;
    int64_t w                = width;
    if (channels == C_BLK()) {
        __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
        __m128 xmm8, xmm9, xmm10, xmm11;
        while (w >= unroll_w * 2) {
            w -= unroll_w * 2;
            xmm0 = _mm_loadu_ps(l_dst_trans + 0 * C_BLK());
            xmm1 = _mm_loadu_ps(l_dst_trans + 1 * C_BLK());
            xmm2 = _mm_loadu_ps(l_dst_trans + 2 * C_BLK());
            xmm3 = _mm_loadu_ps(l_dst_trans + 3 * C_BLK());
            xmm4 = _mm_loadu_ps(l_dst_trans + 4 * C_BLK());
            xmm5 = _mm_loadu_ps(l_dst_trans + 5 * C_BLK());
            xmm6 = _mm_loadu_ps(l_dst_trans + 6 * C_BLK());
            xmm7 = _mm_loadu_ps(l_dst_trans + 7 * C_BLK());
            l_dst_trans += 2 * unroll_w * C_BLK();

            TRANSPOSE_4X4_FP32_SSE_MACRO(xmm0, xmm1, xmm2, xmm3, xmm8, xmm9, xmm10, xmm11);
            TRANSPOSE_4X4_FP32_SSE_MACRO(xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11);

            _mm_storeu_ps(l_dst + 0 * unroll_w + 0 * dst_c_stride, xmm0);
            _mm_storeu_ps(l_dst + 0 * unroll_w + 1 * dst_c_stride, xmm1);
            _mm_storeu_ps(l_dst + 0 * unroll_w + 2 * dst_c_stride, xmm2);
            _mm_storeu_ps(l_dst + 0 * unroll_w + 3 * dst_c_stride, xmm3);
            _mm_storeu_ps(l_dst + 1 * unroll_w + 0 * dst_c_stride, xmm4);
            _mm_storeu_ps(l_dst + 1 * unroll_w + 1 * dst_c_stride, xmm5);
            _mm_storeu_ps(l_dst + 1 * unroll_w + 2 * dst_c_stride, xmm6);
            _mm_storeu_ps(l_dst + 1 * unroll_w + 3 * dst_c_stride, xmm7);
            l_dst += 2 * unroll_w;
        }
        if (w & 4) {
            xmm0 = _mm_loadu_ps(l_dst_trans + 0 * C_BLK());
            xmm1 = _mm_loadu_ps(l_dst_trans + 1 * C_BLK());
            xmm2 = _mm_loadu_ps(l_dst_trans + 2 * C_BLK());
            xmm3 = _mm_loadu_ps(l_dst_trans + 3 * C_BLK());
            l_dst_trans += 1 * unroll_w * C_BLK();

            TRANSPOSE_4X4_FP32_SSE_MACRO(xmm0, xmm1, xmm2, xmm3, xmm8, xmm9, xmm10, xmm11);

            _mm_storeu_ps(l_dst + 0 * unroll_w + 0 * dst_c_stride, xmm0);
            _mm_storeu_ps(l_dst + 0 * unroll_w + 1 * dst_c_stride, xmm1);
            _mm_storeu_ps(l_dst + 0 * unroll_w + 2 * dst_c_stride, xmm2);
            _mm_storeu_ps(l_dst + 0 * unroll_w + 3 * dst_c_stride, xmm3);
            l_dst += 1 * unroll_w;
        }
        if (w & 2) {
            l_dst[0 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 0];
            l_dst[0 * dst_c_stride + 1] = l_dst_trans[1 * C_BLK() + 0];
            l_dst[1 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 1];
            l_dst[1 * dst_c_stride + 1] = l_dst_trans[1 * C_BLK() + 1];
            l_dst[2 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 2];
            l_dst[2 * dst_c_stride + 1] = l_dst_trans[1 * C_BLK() + 2];
            l_dst[3 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 3];
            l_dst[3 * dst_c_stride + 1] = l_dst_trans[1 * C_BLK() + 3];
            l_dst_trans += 2 * C_BLK();
            l_dst += 2;
        }
        if (w & 1) {
            l_dst[0 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 0];
            l_dst[1 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 1];
            l_dst[2 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 2];
            l_dst[3 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 3];
        }
    } else if (channels == 3) {
        while (w >= unroll_w) {
            w -= unroll_w;
            l_dst[0 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 0];
            l_dst[0 * dst_c_stride + 1] = l_dst_trans[1 * C_BLK() + 0];
            l_dst[0 * dst_c_stride + 2] = l_dst_trans[2 * C_BLK() + 0];
            l_dst[0 * dst_c_stride + 3] = l_dst_trans[3 * C_BLK() + 0];
            l_dst[1 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 1];
            l_dst[1 * dst_c_stride + 1] = l_dst_trans[1 * C_BLK() + 1];
            l_dst[1 * dst_c_stride + 2] = l_dst_trans[2 * C_BLK() + 1];
            l_dst[1 * dst_c_stride + 3] = l_dst_trans[3 * C_BLK() + 1];
            l_dst[2 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 2];
            l_dst[2 * dst_c_stride + 1] = l_dst_trans[1 * C_BLK() + 2];
            l_dst[2 * dst_c_stride + 2] = l_dst_trans[2 * C_BLK() + 2];
            l_dst[2 * dst_c_stride + 3] = l_dst_trans[3 * C_BLK() + 2];
            l_dst_trans += unroll_w * C_BLK();
            l_dst += unroll_w;
        }
        if (w & 2) {
            l_dst[0 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 0];
            l_dst[0 * dst_c_stride + 1] = l_dst_trans[1 * C_BLK() + 0];
            l_dst[1 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 1];
            l_dst[1 * dst_c_stride + 1] = l_dst_trans[1 * C_BLK() + 1];
            l_dst[2 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 2];
            l_dst[2 * dst_c_stride + 1] = l_dst_trans[1 * C_BLK() + 2];
            l_dst_trans += 2 * C_BLK();
            l_dst += 2;
        }
        if (w & 1) {
            l_dst[0 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 0];
            l_dst[1 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 1];
            l_dst[2 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 2];
        }
    } else if (channels == 2) {
        while (w >= unroll_w) {
            w -= unroll_w;
            l_dst[0 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 0];
            l_dst[0 * dst_c_stride + 1] = l_dst_trans[1 * C_BLK() + 0];
            l_dst[0 * dst_c_stride + 2] = l_dst_trans[2 * C_BLK() + 0];
            l_dst[0 * dst_c_stride + 3] = l_dst_trans[3 * C_BLK() + 0];
            l_dst[1 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 1];
            l_dst[1 * dst_c_stride + 1] = l_dst_trans[1 * C_BLK() + 1];
            l_dst[1 * dst_c_stride + 2] = l_dst_trans[2 * C_BLK() + 1];
            l_dst[1 * dst_c_stride + 3] = l_dst_trans[3 * C_BLK() + 1];
            l_dst_trans += unroll_w * C_BLK();
            l_dst += unroll_w;
        }
        if (w & 2) {
            l_dst[0 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 0];
            l_dst[0 * dst_c_stride + 1] = l_dst_trans[1 * C_BLK() + 0];
            l_dst[1 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 1];
            l_dst[1 * dst_c_stride + 1] = l_dst_trans[1 * C_BLK() + 1];
            l_dst_trans += 2 * C_BLK();
            l_dst += 2;
        }
        if (w & 1) {
            l_dst[0 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 0];
            l_dst[1 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 1];
        }
    } else {
        while (w >= unroll_w) {
            w -= unroll_w;
            l_dst[0 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 0];
            l_dst[0 * dst_c_stride + 1] = l_dst_trans[1 * C_BLK() + 0];
            l_dst[0 * dst_c_stride + 2] = l_dst_trans[2 * C_BLK() + 0];
            l_dst[0 * dst_c_stride + 3] = l_dst_trans[3 * C_BLK() + 0];
            l_dst_trans += unroll_w * C_BLK();
            l_dst += unroll_w;
        }
        if (w & 2) {
            l_dst[0 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 0];
            l_dst[0 * dst_c_stride + 1] = l_dst_trans[1 * C_BLK() + 0];
            l_dst_trans += 2 * C_BLK();
            l_dst += 2;
        }
        if (w & 1) {
            l_dst[0 * dst_c_stride + 0] = l_dst_trans[0 * C_BLK() + 0];
        }
    }
}

static void pooling2d_fp32_sse_src_trans(
    const float *src,
    const averagepool2d_param *param,
    const int64_t ob,
    const int64_t oc,
    const int64_t c_eff,
    float *dst)
{
    const int64_t &channels = param->channels;
    const int64_t &src_h    = param->src_h;
    const int64_t &src_w    = param->src_w;
    const int64_t src_hw    = src_h * src_w;

    float *l_src_trans    = dst;
    const float *base_src = src + ob * channels * src_hw + oc * src_hw;

    for (int64_t ih = 0; ih < src_h; ++ih) {
        const int64_t unroll_w = 4;
        int64_t iw             = src_w;
        if (c_eff == C_BLK()) {
            while (iw >= unroll_w * 2) {
                iw -= 2 * unroll_w;
                transpose_4x4_fp32_sse(base_src + 0 * unroll_w, src_hw, C_BLK(), l_src_trans + 0 * unroll_w * C_BLK());
                transpose_4x4_fp32_sse(base_src + 1 * unroll_w, src_hw, C_BLK(), l_src_trans + 1 * unroll_w * C_BLK());
                base_src += 2 * unroll_w;
                l_src_trans += 2 * unroll_w * C_BLK();
            }
            if (iw & 4) {
                transpose_4x4_fp32_sse(base_src + 0 * unroll_w, src_hw, C_BLK(), l_src_trans + 0 * unroll_w * C_BLK());
                iw -= 1 * unroll_w;
                base_src += 1 * unroll_w;
                l_src_trans += 1 * unroll_w * C_BLK();
            }
            if (iw & 2) {
                l_src_trans[0 + 0 * C_BLK()] = base_src[0 * src_hw + 0];
                l_src_trans[1 + 0 * C_BLK()] = base_src[1 * src_hw + 0];
                l_src_trans[2 + 0 * C_BLK()] = base_src[2 * src_hw + 0];
                l_src_trans[3 + 0 * C_BLK()] = base_src[3 * src_hw + 0];
                l_src_trans[0 + 1 * C_BLK()] = base_src[0 * src_hw + 1];
                l_src_trans[1 + 1 * C_BLK()] = base_src[1 * src_hw + 1];
                l_src_trans[2 + 1 * C_BLK()] = base_src[2 * src_hw + 1];
                l_src_trans[3 + 1 * C_BLK()] = base_src[3 * src_hw + 1];
                base_src += 2;
                l_src_trans += 2 * C_BLK();
            }
            if (iw & 1) {
                l_src_trans[0 + 0 * C_BLK()] = base_src[0 * src_hw + 0];
                l_src_trans[1 + 0 * C_BLK()] = base_src[1 * src_hw + 0];
                l_src_trans[2 + 0 * C_BLK()] = base_src[2 * src_hw + 0];
                l_src_trans[3 + 0 * C_BLK()] = base_src[3 * src_hw + 0];
                base_src += 1;
                l_src_trans += 1 * C_BLK();
            }
        } else if (c_eff == 3) {
            while (iw >= unroll_w) {
                l_src_trans[0 + 0 * C_BLK()] = base_src[0 * src_hw + 0];
                l_src_trans[1 + 0 * C_BLK()] = base_src[1 * src_hw + 0];
                l_src_trans[2 + 0 * C_BLK()] = base_src[2 * src_hw + 0];
                l_src_trans[0 + 1 * C_BLK()] = base_src[0 * src_hw + 1];
                l_src_trans[1 + 1 * C_BLK()] = base_src[1 * src_hw + 1];
                l_src_trans[2 + 1 * C_BLK()] = base_src[2 * src_hw + 1];
                l_src_trans[0 + 2 * C_BLK()] = base_src[0 * src_hw + 2];
                l_src_trans[1 + 2 * C_BLK()] = base_src[1 * src_hw + 2];
                l_src_trans[2 + 2 * C_BLK()] = base_src[2 * src_hw + 2];
                l_src_trans[0 + 3 * C_BLK()] = base_src[0 * src_hw + 3];
                l_src_trans[1 + 3 * C_BLK()] = base_src[1 * src_hw + 3];
                l_src_trans[2 + 3 * C_BLK()] = base_src[2 * src_hw + 3];
                iw -= unroll_w;
                base_src += unroll_w;
                l_src_trans += unroll_w * C_BLK();
            }
            if (iw & 2) {
                l_src_trans[0 + 0 * C_BLK()] = base_src[0 * src_hw + 0];
                l_src_trans[1 + 0 * C_BLK()] = base_src[1 * src_hw + 0];
                l_src_trans[2 + 0 * C_BLK()] = base_src[2 * src_hw + 0];
                l_src_trans[0 + 1 * C_BLK()] = base_src[0 * src_hw + 1];
                l_src_trans[1 + 1 * C_BLK()] = base_src[1 * src_hw + 1];
                l_src_trans[2 + 1 * C_BLK()] = base_src[2 * src_hw + 1];
                base_src += 2;
                l_src_trans += 2 * C_BLK();
            }
            if (iw & 1) {
                l_src_trans[0 + 0 * C_BLK()] = base_src[0 * src_hw + 0];
                l_src_trans[1 + 0 * C_BLK()] = base_src[1 * src_hw + 0];
                l_src_trans[2 + 0 * C_BLK()] = base_src[2 * src_hw + 0];
                base_src += 1;
                l_src_trans += 1 * C_BLK();
            }
        } else if (c_eff == 2) {
            while (iw >= unroll_w) {
                l_src_trans[0 + 0 * C_BLK()] = base_src[0 * src_hw + 0];
                l_src_trans[1 + 0 * C_BLK()] = base_src[1 * src_hw + 0];
                l_src_trans[0 + 1 * C_BLK()] = base_src[0 * src_hw + 1];
                l_src_trans[1 + 1 * C_BLK()] = base_src[1 * src_hw + 1];
                l_src_trans[0 + 2 * C_BLK()] = base_src[0 * src_hw + 2];
                l_src_trans[1 + 2 * C_BLK()] = base_src[1 * src_hw + 2];
                l_src_trans[0 + 3 * C_BLK()] = base_src[0 * src_hw + 3];
                l_src_trans[1 + 3 * C_BLK()] = base_src[1 * src_hw + 3];
                iw -= unroll_w;
                base_src += unroll_w;
                l_src_trans += unroll_w * C_BLK();
            }
            if (iw & 2) {
                l_src_trans[0 + 0 * C_BLK()] = base_src[0 * src_hw + 0];
                l_src_trans[1 + 0 * C_BLK()] = base_src[1 * src_hw + 0];
                l_src_trans[0 + 1 * C_BLK()] = base_src[0 * src_hw + 1];
                l_src_trans[1 + 1 * C_BLK()] = base_src[1 * src_hw + 1];
                base_src += 2;
                l_src_trans += 2 * C_BLK();
            }
            if (iw & 1) {
                l_src_trans[0 + 0 * C_BLK()] = base_src[0 * src_hw + 0];
                l_src_trans[1 + 0 * C_BLK()] = base_src[1 * src_hw + 0];
                base_src += 1;
                l_src_trans += 1 * C_BLK();
            }
        } else {
            while (iw >= unroll_w) {
                l_src_trans[0 + 0 * C_BLK()] = base_src[0 * src_hw + 0];
                l_src_trans[0 + 1 * C_BLK()] = base_src[0 * src_hw + 1];
                l_src_trans[0 + 2 * C_BLK()] = base_src[0 * src_hw + 2];
                l_src_trans[0 + 3 * C_BLK()] = base_src[0 * src_hw + 3];
                iw -= unroll_w;
                base_src += unroll_w;
                l_src_trans += unroll_w * C_BLK();
            }
            if (iw & 2) {
                l_src_trans[0 + 0 * C_BLK()] = base_src[0 * src_hw + 0];
                l_src_trans[0 + 1 * C_BLK()] = base_src[0 * src_hw + 1];
                base_src += 2;
                l_src_trans += 2 * C_BLK();
            }
            if (iw & 1) {
                l_src_trans[0 + 0 * C_BLK()] = base_src[0 * src_hw + 0];
                base_src += 1;
                l_src_trans += 1 * C_BLK();
            }
        }
    }
}

static inline void averagepool2d_ndarray_border_fp32_sse_n4cx_impl(
    const float *src,
    const averagepool2d_param *param,
    const int64_t ow,
    const int64_t ih_start,
    const int64_t ih_end,
    const int64_t padding_ih_len,
    int32_t pooling_mode,
    bool ceil_mode,
    float *dst)
{
    const int64_t &kernel_w = param->kernel_w;
    const int64_t &stride_w = param->stride_w;
    const int64_t &pad_w    = param->pad_w;
    const int64_t &src_w    = param->src_w;

    const int64_t padded_iwstart = ow * stride_w - pad_w;
    const int64_t padded_iwend   = ceil_mode ? padded_iwstart + kernel_w : min<int64_t>(padded_iwstart + kernel_w, src_w + pad_w);

    const int64_t iw_start = max<int64_t>(padded_iwstart, 0);
    const int64_t iw_end   = min<int64_t>(padded_iwend, src_w);

    int64_t pool_len = 0;

    if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE) {
        pool_len = (ih_end - ih_start) * (iw_end - iw_start);
    } else if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE) {
        pool_len = padding_ih_len * (padded_iwend - padded_iwstart);
    }

    __m128 zmm00      = _mm_setzero_ps();
    __m128 vave_coeff = _mm_set1_ps(1.0f / pool_len);

    for (int64_t ih = ih_start; ih < ih_end; ++ih) {
        for (int64_t iw = iw_start; iw < iw_end; ++iw) {
            zmm00 = _mm_add_ps(zmm00, _mm_loadu_ps(src + (ih * src_w + iw) * C_BLK()));
        }
    }

    _mm_storeu_ps(dst + ow * C_BLK(), _mm_mul_ps(zmm00, vave_coeff));
}

template <int64_t spec_stride_w, int64_t w_len>
static void averagepool2d_ndarray_1x4_kernel_fp32_sse_n4cx(
    const float *src,
    const averagepool2d_param *param,
    const int64_t ow,
    const int64_t ih_start,
    const int64_t ih_end,
    const int64_t padding_ih_len,
    int32_t pooling_mode,
    bool ceil_mode,
    float *dst)
{
    const int64_t &kernel_w = param->kernel_w;
    const int64_t &pad_w    = param->pad_w;
    const int64_t &src_w    = param->src_w;
    const int64_t stride_w  = spec_stride_w ? spec_stride_w : param->stride_w;

    const int64_t padded_iwstart = ow * stride_w - pad_w;
    const int64_t padded_iwend   = ceil_mode ? padded_iwstart + kernel_w : min<int64_t>(padded_iwstart + kernel_w, src_w + pad_w);

    const int64_t padded_iwstart_1 = padded_iwstart + stride_w;
    const int64_t padded_iwend_1   = ceil_mode ? padded_iwstart_1 + kernel_w : min<int64_t>(padded_iwstart_1 + kernel_w, src_w + pad_w);

    const int64_t padded_iwstart_2 = padded_iwstart_1 + stride_w;
    const int64_t padded_iwend_2   = ceil_mode ? padded_iwstart_2 + kernel_w : min<int64_t>(padded_iwstart_2 + kernel_w, src_w + pad_w);

    const int64_t padded_iwstart_3 = padded_iwstart_2 + stride_w;
    const int64_t padded_iwend_3   = ceil_mode ? padded_iwstart_3 + kernel_w : min<int64_t>(padded_iwstart_3 + kernel_w, src_w + pad_w);

    const int64_t iw_start = max<int64_t>(padded_iwstart, 0);
    const int64_t iw_end   = min<int64_t>(padded_iwend, src_w);

    const int64_t iw_start_1 = max<int64_t>(padded_iwstart_1, 0);
    const int64_t iw_end_1   = min<int64_t>(padded_iwend_1, src_w);

    const int64_t iw_start_2 = max<int64_t>(padded_iwstart_2, 0);
    const int64_t iw_end_2   = min<int64_t>(padded_iwend_2, src_w);

    const int64_t iw_start_3 = max<int64_t>(padded_iwstart_3, 0);
    const int64_t iw_end_3   = min<int64_t>(padded_iwend_3, src_w);

    float *p_dst = dst + ow * C_BLK();

    int64_t pool_len   = 0;
    int64_t pool_len_1 = 0;
    int64_t pool_len_2 = 0;
    int64_t pool_len_3 = 0;
    if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE) {
        pool_len   = (ih_end - ih_start) * (iw_end - iw_start);
        pool_len_1 = (ih_end - ih_start) * (iw_end_1 - iw_start_1);
        pool_len_2 = (ih_end - ih_start) * (iw_end_2 - iw_start_2);
        pool_len_3 = (ih_end - ih_start) * (iw_end_3 - iw_start_3);
    } else if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE) {
        pool_len   = padding_ih_len * (padded_iwend - padded_iwstart);
        pool_len_1 = padding_ih_len * (padded_iwend_1 - padded_iwstart_1);
        pool_len_2 = padding_ih_len * (padded_iwend_2 - padded_iwstart_2);
        pool_len_3 = padding_ih_len * (padded_iwend_3 - padded_iwstart_3);
    }

    __m128 zmm00, zmm01, zmm02, zmm03;
    __m128 vave_coeff   = _mm_set1_ps(1.0f / pool_len);
    __m128 vave_coeff_1 = _mm_set1_ps(1.0f / pool_len_1);
    __m128 vave_coeff_2 = _mm_set1_ps(1.0f / pool_len_2);
    __m128 vave_coeff_3 = _mm_set1_ps(1.0f / pool_len_3);
    if (w_len >= 1) zmm00 = _mm_setzero_ps();
    if (w_len >= 2) zmm01 = zmm00;
    if (w_len >= 3) zmm02 = zmm00;
    if (w_len >= 4) zmm03 = zmm00;

    for (int64_t ih = ih_start; ih < ih_end; ++ih) {
        for (int64_t iw = iw_start; iw < iw_end; ++iw) {
            const float *p_src = src + (ih * src_w + iw) * C_BLK();
            if (w_len >= 1) zmm00 = _mm_add_ps(zmm00, _mm_loadu_ps(p_src + 0 * stride_w * C_BLK()));
            if (w_len >= 2) zmm01 = _mm_add_ps(zmm01, _mm_loadu_ps(p_src + 1 * stride_w * C_BLK()));
            if (w_len >= 3) zmm02 = _mm_add_ps(zmm02, _mm_loadu_ps(p_src + 2 * stride_w * C_BLK()));
            if (w_len >= 4) zmm03 = _mm_add_ps(zmm03, _mm_loadu_ps(p_src + 3 * stride_w * C_BLK()));
        }
    }

    if (w_len >= 1) _mm_storeu_ps(p_dst + 0 * C_BLK(), _mm_mul_ps(zmm00, vave_coeff));
    if (w_len >= 2) _mm_storeu_ps(p_dst + 1 * C_BLK(), _mm_mul_ps(zmm01, vave_coeff_1));
    if (w_len >= 3) _mm_storeu_ps(p_dst + 2 * C_BLK(), _mm_mul_ps(zmm02, vave_coeff_2));
    if (w_len >= 4) _mm_storeu_ps(p_dst + 3 * C_BLK(), _mm_mul_ps(zmm03, vave_coeff_3));
}

typedef void (*averagepool2d_ndarray_kernel_fp32_sse_n4cx_funct)(const float *, const averagepool2d_param *, const int64_t, const int64_t, const int64_t, const int64_t, int32_t, bool, float *);
static const averagepool2d_ndarray_kernel_fp32_sse_n4cx_funct averagepool2d_ndarray_1x4_kernel_n4cx_func_table[3][5]{
    {
        averagepool2d_ndarray_1x4_kernel_fp32_sse_n4cx<0, 0>,
        averagepool2d_ndarray_1x4_kernel_fp32_sse_n4cx<0, 1>,
        averagepool2d_ndarray_1x4_kernel_fp32_sse_n4cx<0, 2>,
        averagepool2d_ndarray_1x4_kernel_fp32_sse_n4cx<0, 3>,
        averagepool2d_ndarray_1x4_kernel_fp32_sse_n4cx<0, 4>,
    },
    {
        averagepool2d_ndarray_1x4_kernel_fp32_sse_n4cx<1, 0>,
        averagepool2d_ndarray_1x4_kernel_fp32_sse_n4cx<1, 1>,
        averagepool2d_ndarray_1x4_kernel_fp32_sse_n4cx<1, 2>,
        averagepool2d_ndarray_1x4_kernel_fp32_sse_n4cx<1, 3>,
        averagepool2d_ndarray_1x4_kernel_fp32_sse_n4cx<1, 4>,
    },
    {
        averagepool2d_ndarray_1x4_kernel_fp32_sse_n4cx<2, 0>,
        averagepool2d_ndarray_1x4_kernel_fp32_sse_n4cx<2, 1>,
        averagepool2d_ndarray_1x4_kernel_fp32_sse_n4cx<2, 2>,
        averagepool2d_ndarray_1x4_kernel_fp32_sse_n4cx<2, 3>,
        averagepool2d_ndarray_1x4_kernel_fp32_sse_n4cx<2, 4>,
    },

};

template <int32_t pooling_mode, bool ceil_mode>
static ppl::common::RetCode averagepool2d_ndarray_fp32_sse_n4cx_impl(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w,
    void *temp_buffer,
    float *dst)
{
    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t src_h    = src_shape->GetDim(2);
    const int64_t src_w    = src_shape->GetDim(3);
    const int64_t dst_h    = dst_shape->GetDim(2);
    const int64_t dst_w    = dst_shape->GetDim(3);

    const int64_t dst_hw        = dst_h * dst_w;
    const int64_t src_trans_len = src_h * (src_w + pad_w) * C_BLK();
    const int64_t dst_len       = dst_w * C_BLK();
    const int64_t thread_buf    = src_trans_len + dst_len;
    const int64_t pad_c         = round_up(channels, C_BLK());

    const int64_t dst_kernel_start_w = max<int64_t>((pad_w + stride_w - 1) / stride_w, 0);
    const int64_t dst_kernel_end_w   = min<int64_t>((src_w + pad_w - kernel_w) / stride_w + 1, dst_w);

    const int64_t stride_w_select = stride_w > 2 ? 0 : stride_w;

    const averagepool2d_param param = {kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, batch, channels, src_h, src_w, dst_h, dst_w};

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t bc = 0; bc < batch * pad_c; bc += C_BLK()) {
        float *tmpbuf = reinterpret_cast<float *>(temp_buffer) + PPL_OMP_THREAD_ID() * thread_buf;
        float *dstbuf = tmpbuf + src_trans_len;

        const int64_t b = bc / pad_c;
        const int64_t c = bc % pad_c;
        float *base_dst = dst + b * channels * dst_hw + c * dst_hw;
        int64_t c_eff   = min<int64_t>(C_BLK(), channels - c);

        pooling2d_fp32_sse_src_trans(src, &param, b, c, c_eff, tmpbuf);

        // DO POOLING
        const float *p_src_c = tmpbuf;
        float *p_dst_c       = dstbuf;

        for (int64_t oh = 0; oh < dst_h; oh++) {
            const int64_t padded_ihstart = oh * stride_h - pad_h;
            const int64_t padded_ihend   = ceil_mode ? padded_ihstart + kernel_h : min<int64_t>(padded_ihstart + kernel_h, src_h + pad_h);
            const int64_t ih_start       = max<int64_t>(padded_ihstart, 0);
            const int64_t ih_end         = min<int64_t>(padded_ihend, src_h);

            if (dst_kernel_start_w >= dst_kernel_end_w) {
                for (int64_t ow = 0; ow < dst_w; ow++) {
                    averagepool2d_ndarray_border_fp32_sse_n4cx_impl(p_src_c, &param, ow, ih_start, ih_end, padded_ihend - padded_ihstart, pooling_mode, ceil_mode, p_dst_c);
                }
            } else {
                int64_t ow = 0;
                for (; ow < dst_kernel_start_w; ow++) {
                    averagepool2d_ndarray_border_fp32_sse_n4cx_impl(p_src_c, &param, ow, ih_start, ih_end, padded_ihend - padded_ihstart, pooling_mode, ceil_mode, p_dst_c);
                }
                for (; ow + C_BLK() <= dst_kernel_end_w; ow += C_BLK()) {
                    averagepool2d_ndarray_1x4_kernel_n4cx_func_table[stride_w_select][C_BLK()](p_src_c, &param, ow, ih_start, ih_end, padded_ihend - padded_ihstart, pooling_mode, ceil_mode, p_dst_c);
                }
                if (ow < dst_kernel_end_w) {
                    averagepool2d_ndarray_1x4_kernel_n4cx_func_table[stride_w_select][dst_kernel_end_w - ow](p_src_c, &param, ow, ih_start, ih_end, padded_ihend - padded_ihstart, pooling_mode, ceil_mode, p_dst_c);
                    ow = dst_kernel_end_w;
                }
                for (; ow < dst_w; ow++) {
                    averagepool2d_ndarray_border_fp32_sse_n4cx_impl(p_src_c, &param, ow, ih_start, ih_end, padded_ihend - padded_ihstart, pooling_mode, ceil_mode, p_dst_c);
                }
            }
            pooling2d_fp32_sse_dst_trans(p_dst_c, dst_w, c_eff, dst_hw, base_dst);
            base_dst += dst_w;
        }
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode averagepool2d_ndarray_normal_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t pooling_mode,
    const int64_t ceil_mode,
    void *temp_buffer,
    float *dst)
{
    if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE) {
        if (ceil_mode) {
            return averagepool2d_ndarray_fp32_sse_n4cx_impl<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, true>(src_shape, dst_shape, src, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, temp_buffer, dst);
        } else {
            return averagepool2d_ndarray_fp32_sse_n4cx_impl<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, false>(src_shape, dst_shape, src, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, temp_buffer, dst);
        }
    } else if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE) {
        if (ceil_mode) {
            return averagepool2d_ndarray_fp32_sse_n4cx_impl<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, true>(src_shape, dst_shape, src, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, temp_buffer, dst);
        } else {
            return averagepool2d_ndarray_fp32_sse_n4cx_impl<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, false>(src_shape, dst_shape, src, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, temp_buffer, dst);
        }
    }

    return ppl::common::RC_INVALID_VALUE;
}

}}}; // namespace ppl::kernel::x86
