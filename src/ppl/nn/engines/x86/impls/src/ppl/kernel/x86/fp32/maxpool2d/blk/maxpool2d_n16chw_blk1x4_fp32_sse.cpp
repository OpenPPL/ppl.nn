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
#include <float.h>
#include <string.h> // for memset

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/maxpool2d/maxpool2d_common.h"

namespace ppl { namespace kernel { namespace x86 {

#define STRIDE_W_OPT()           3
#define POOLING_DST_W()          4
#define POOLING_CHANNELS_BLOCK() 16
#define SIMD_W()                 4

template <int64_t spec_stride_w, int64_t w_len>
static void maxpool2d_n16chw_1x4_kernel_fp32_sse(
    const float *src,
    const maxpool2d_param *param,
    const int64_t oh,
    const int64_t ow,
    const int64_t ihstart,
    const int64_t ihend,
    float *dst)
{
    const int32_t &kernel_w = param->kernel_w;
    const int32_t &pad_w    = param->pad_w;

    const int32_t &src_w = param->src_w;
    const int32_t &dst_w = param->dst_w;

    const int64_t stride_w = spec_stride_w ? spec_stride_w : param->stride_w;

    const int64_t iwstart = ow * stride_w - pad_w; // will always >= 0
    const int64_t iwend   = iwstart + kernel_w; // will always < src_w

    const int64_t c_blk_len = POOLING_CHANNELS_BLOCK();

    __m128 xmm00, xmm01, xmm02, xmm03;
    __m128 xmm04, xmm05, xmm06, xmm07;
    __m128 xmm08, xmm09, xmm10, xmm11;
    __m128 xmm12, xmm13, xmm14, xmm15;
    if (w_len >= 1) xmm00 = _mm_set1_ps(-FLT_MAX);
    if (w_len >= 1) xmm01 = xmm00;
    if (w_len >= 1) xmm02 = xmm00;
    if (w_len >= 1) xmm03 = xmm00;
    if (w_len >= 2) xmm04 = xmm00;
    if (w_len >= 2) xmm05 = xmm00;
    if (w_len >= 2) xmm06 = xmm00;
    if (w_len >= 2) xmm07 = xmm00;
    if (w_len >= 3) xmm08 = xmm00;
    if (w_len >= 3) xmm09 = xmm00;
    if (w_len >= 3) xmm10 = xmm00;
    if (w_len >= 3) xmm11 = xmm00;
    if (w_len >= 4) xmm12 = xmm00;
    if (w_len >= 4) xmm13 = xmm00;
    if (w_len >= 4) xmm14 = xmm00;
    if (w_len >= 4) xmm15 = xmm00;

    for (int64_t ih = ihstart; ih < ihend; ++ih) {
        for (int64_t iw = iwstart; iw < iwend; ++iw) {
            const float *p_src = src + (ih * src_w + iw) * c_blk_len;
            if (w_len >= 1) xmm00 = _mm_max_ps(xmm00, _mm_loadu_ps(p_src + 0 * stride_w * c_blk_len + 0 * SIMD_W()));
            if (w_len >= 1) xmm01 = _mm_max_ps(xmm01, _mm_loadu_ps(p_src + 0 * stride_w * c_blk_len + 1 * SIMD_W()));
            if (w_len >= 1) xmm02 = _mm_max_ps(xmm02, _mm_loadu_ps(p_src + 0 * stride_w * c_blk_len + 2 * SIMD_W()));
            if (w_len >= 1) xmm03 = _mm_max_ps(xmm03, _mm_loadu_ps(p_src + 0 * stride_w * c_blk_len + 3 * SIMD_W()));
            if (w_len >= 2) xmm04 = _mm_max_ps(xmm04, _mm_loadu_ps(p_src + 1 * stride_w * c_blk_len + 0 * SIMD_W()));
            if (w_len >= 2) xmm05 = _mm_max_ps(xmm05, _mm_loadu_ps(p_src + 1 * stride_w * c_blk_len + 1 * SIMD_W()));
            if (w_len >= 2) xmm06 = _mm_max_ps(xmm06, _mm_loadu_ps(p_src + 1 * stride_w * c_blk_len + 2 * SIMD_W()));
            if (w_len >= 2) xmm07 = _mm_max_ps(xmm07, _mm_loadu_ps(p_src + 1 * stride_w * c_blk_len + 3 * SIMD_W()));
            if (w_len >= 3) xmm08 = _mm_max_ps(xmm08, _mm_loadu_ps(p_src + 2 * stride_w * c_blk_len + 0 * SIMD_W()));
            if (w_len >= 3) xmm09 = _mm_max_ps(xmm09, _mm_loadu_ps(p_src + 2 * stride_w * c_blk_len + 1 * SIMD_W()));
            if (w_len >= 3) xmm10 = _mm_max_ps(xmm10, _mm_loadu_ps(p_src + 2 * stride_w * c_blk_len + 2 * SIMD_W()));
            if (w_len >= 3) xmm11 = _mm_max_ps(xmm11, _mm_loadu_ps(p_src + 2 * stride_w * c_blk_len + 3 * SIMD_W()));
            if (w_len >= 4) xmm12 = _mm_max_ps(xmm12, _mm_loadu_ps(p_src + 3 * stride_w * c_blk_len + 0 * SIMD_W()));
            if (w_len >= 4) xmm13 = _mm_max_ps(xmm13, _mm_loadu_ps(p_src + 3 * stride_w * c_blk_len + 1 * SIMD_W()));
            if (w_len >= 4) xmm14 = _mm_max_ps(xmm14, _mm_loadu_ps(p_src + 3 * stride_w * c_blk_len + 2 * SIMD_W()));
            if (w_len >= 4) xmm15 = _mm_max_ps(xmm15, _mm_loadu_ps(p_src + 3 * stride_w * c_blk_len + 3 * SIMD_W()));
        }
    }

    float *p_dst = dst + (oh * dst_w + ow) * c_blk_len;
    if (w_len >= 1) _mm_storeu_ps(p_dst + 0 * c_blk_len + 0 * SIMD_W(), xmm00);
    if (w_len >= 1) _mm_storeu_ps(p_dst + 0 * c_blk_len + 1 * SIMD_W(), xmm01);
    if (w_len >= 1) _mm_storeu_ps(p_dst + 0 * c_blk_len + 2 * SIMD_W(), xmm02);
    if (w_len >= 1) _mm_storeu_ps(p_dst + 0 * c_blk_len + 3 * SIMD_W(), xmm03);
    if (w_len >= 2) _mm_storeu_ps(p_dst + 1 * c_blk_len + 0 * SIMD_W(), xmm04);
    if (w_len >= 2) _mm_storeu_ps(p_dst + 1 * c_blk_len + 1 * SIMD_W(), xmm05);
    if (w_len >= 2) _mm_storeu_ps(p_dst + 1 * c_blk_len + 2 * SIMD_W(), xmm06);
    if (w_len >= 2) _mm_storeu_ps(p_dst + 1 * c_blk_len + 3 * SIMD_W(), xmm07);
    if (w_len >= 3) _mm_storeu_ps(p_dst + 2 * c_blk_len + 0 * SIMD_W(), xmm08);
    if (w_len >= 3) _mm_storeu_ps(p_dst + 2 * c_blk_len + 1 * SIMD_W(), xmm09);
    if (w_len >= 3) _mm_storeu_ps(p_dst + 2 * c_blk_len + 2 * SIMD_W(), xmm10);
    if (w_len >= 3) _mm_storeu_ps(p_dst + 2 * c_blk_len + 3 * SIMD_W(), xmm11);
    if (w_len >= 4) _mm_storeu_ps(p_dst + 3 * c_blk_len + 0 * SIMD_W(), xmm12);
    if (w_len >= 4) _mm_storeu_ps(p_dst + 3 * c_blk_len + 1 * SIMD_W(), xmm13);
    if (w_len >= 4) _mm_storeu_ps(p_dst + 3 * c_blk_len + 2 * SIMD_W(), xmm14);
    if (w_len >= 4) _mm_storeu_ps(p_dst + 3 * c_blk_len + 3 * SIMD_W(), xmm15);
}

typedef void (*maxpool2d_n16chw_kernel_fp32_sse_func_t)(const float *, const maxpool2d_param *, const int64_t, const int64_t, const int64_t, const int64_t, float *);
static const maxpool2d_n16chw_kernel_fp32_sse_func_t maxpool2d_n16chw_1x4_kernel_func_table[STRIDE_W_OPT()][POOLING_DST_W() + 1]{
    {
        maxpool2d_n16chw_1x4_kernel_fp32_sse<0, 0>,
        maxpool2d_n16chw_1x4_kernel_fp32_sse<0, 1>,
        maxpool2d_n16chw_1x4_kernel_fp32_sse<0, 2>,
        maxpool2d_n16chw_1x4_kernel_fp32_sse<0, 3>,
        maxpool2d_n16chw_1x4_kernel_fp32_sse<0, 4>,
    },
    {
        maxpool2d_n16chw_1x4_kernel_fp32_sse<1, 0>,
        maxpool2d_n16chw_1x4_kernel_fp32_sse<1, 1>,
        maxpool2d_n16chw_1x4_kernel_fp32_sse<1, 2>,
        maxpool2d_n16chw_1x4_kernel_fp32_sse<1, 3>,
        maxpool2d_n16chw_1x4_kernel_fp32_sse<1, 4>,
    },
    {
        maxpool2d_n16chw_1x4_kernel_fp32_sse<2, 0>,
        maxpool2d_n16chw_1x4_kernel_fp32_sse<2, 1>,
        maxpool2d_n16chw_1x4_kernel_fp32_sse<2, 2>,
        maxpool2d_n16chw_1x4_kernel_fp32_sse<2, 3>,
        maxpool2d_n16chw_1x4_kernel_fp32_sse<2, 4>,
    },
};

static inline void maxpool2d_n16chw_border_fp32_sse(
    const float *src,
    const maxpool2d_param *param,
    const int64_t oh,
    const int64_t ow,
    float *dst)
{
    const int32_t &kernel_h = param->kernel_h;
    const int32_t &kernel_w = param->kernel_w;
    const int32_t &stride_h = param->stride_h;
    const int32_t &stride_w = param->stride_w;
    const int32_t &pad_h    = param->pad_h;
    const int32_t &pad_w    = param->pad_w;

    const int32_t &src_h = param->src_h;
    const int32_t &src_w = param->src_w;
    const int32_t &dst_w = param->dst_w;

    const int64_t c_blk_len = POOLING_CHANNELS_BLOCK();

    const int64_t pre_ihstart = oh * stride_h - pad_h;
    const int64_t ihstart     = max<int64_t>(pre_ihstart, 0);
    const int64_t ihend       = min<int64_t>(pre_ihstart + kernel_h, src_h);
    const int64_t pre_iwstart = ow * stride_w - pad_w;
    const int64_t iwstart     = max<int64_t>(pre_iwstart, 0);
    const int64_t iwend       = min<int64_t>(pre_iwstart + kernel_w, src_w);

    if (ihstart >= ihend || iwstart >= iwend) {
        _mm_storeu_ps(dst + (oh * dst_w + ow) * c_blk_len + 0 * SIMD_W(), _mm_setzero_ps());
        _mm_storeu_ps(dst + (oh * dst_w + ow) * c_blk_len + 1 * SIMD_W(), _mm_setzero_ps());
        _mm_storeu_ps(dst + (oh * dst_w + ow) * c_blk_len + 2 * SIMD_W(), _mm_setzero_ps());
        _mm_storeu_ps(dst + (oh * dst_w + ow) * c_blk_len + 3 * SIMD_W(), _mm_setzero_ps());
    } else {
        __m128 v_max_val0 = _mm_set1_ps(-FLT_MAX);
        __m128 v_max_val1 = v_max_val0;
        __m128 v_max_val2 = v_max_val0;
        __m128 v_max_val3 = v_max_val0;
        for (int64_t ih = ihstart; ih < ihend; ++ih) {
            for (int64_t iw = iwstart; iw < iwend; ++iw) {
                __m128 v_src_val0 = _mm_loadu_ps(src + (ih * src_w + iw) * c_blk_len + 0 * SIMD_W());
                __m128 v_src_val1 = _mm_loadu_ps(src + (ih * src_w + iw) * c_blk_len + 1 * SIMD_W());
                __m128 v_src_val2 = _mm_loadu_ps(src + (ih * src_w + iw) * c_blk_len + 2 * SIMD_W());
                __m128 v_src_val3 = _mm_loadu_ps(src + (ih * src_w + iw) * c_blk_len + 3 * SIMD_W());
                v_max_val0        = _mm_max_ps(v_max_val0, v_src_val0);
                v_max_val1        = _mm_max_ps(v_max_val1, v_src_val1);
                v_max_val2        = _mm_max_ps(v_max_val2, v_src_val2);
                v_max_val3        = _mm_max_ps(v_max_val3, v_src_val3);
            }
        }
        _mm_storeu_ps(dst + (oh * dst_w + ow) * c_blk_len + 0 * SIMD_W(), v_max_val0);
        _mm_storeu_ps(dst + (oh * dst_w + ow) * c_blk_len + 1 * SIMD_W(), v_max_val1);
        _mm_storeu_ps(dst + (oh * dst_w + ow) * c_blk_len + 2 * SIMD_W(), v_max_val2);
        _mm_storeu_ps(dst + (oh * dst_w + ow) * c_blk_len + 3 * SIMD_W(), v_max_val3);
    }
}

ppl::common::RetCode maxpool2d_n16chw_blk1x4_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    float *dst)
{
    const int32_t batch    = src_shape->GetDim(0);
    const int32_t channels = src_shape->GetDim(1);
    const int32_t src_h    = src_shape->GetDim(2);
    const int32_t src_w    = src_shape->GetDim(3);
    const int32_t dst_h    = dst_shape->GetDim(2);
    const int32_t dst_w    = dst_shape->GetDim(3);

    const maxpool2d_param param = {kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, batch, channels, src_h, src_w, dst_h, dst_w};

    const int64_t c_blk_len          = POOLING_CHANNELS_BLOCK();
    const int64_t padded_c           = round_up(channels, c_blk_len);
    const int64_t dst_kernel_start_w = max<int64_t>((pad_w + stride_w - 1) / stride_w, 0);
    const int64_t dst_kernel_end_w   = min<int64_t>((src_w + pad_w - kernel_w) / stride_w + 1, dst_w);

    const int64_t stride_w_select = stride_w > 2 ? 0 : stride_w;

    if (dst_kernel_start_w >= dst_kernel_end_w) { // all output need padding input
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t bc = 0; bc < batch * padded_c; bc += c_blk_len) {
            const float *p_src = src + bc * src_h * src_w;
            float *p_dst       = dst + bc * dst_h * dst_w;
            for (int64_t oh = 0; oh < dst_h; ++oh) {
                for (int64_t ow = 0; ow < dst_w; ++ow) {
                    maxpool2d_n16chw_border_fp32_sse(p_src, &param, oh, ow, p_dst);
                }
            }
        }
        return ppl::common::RC_SUCCESS;
    }

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
    PRAGMA_OMP_PARALLEL_FOR()
#endif
    for (int64_t bc = 0; bc < batch * padded_c; bc += c_blk_len) {
        for (int64_t oh = 0; oh < dst_h; ++oh) {
            const float *p_src = src + bc * src_h * src_w;
            float *p_dst       = dst + bc * dst_h * dst_w;

            const int64_t pre_ihstart = oh * stride_h - pad_h;
            const int64_t ihstart     = max<int64_t>(pre_ihstart, 0);
            const int64_t ihend       = min<int64_t>(pre_ihstart + kernel_h, src_h);
            if (ihstart >= ihend) { // all input lines are padding lines
                memset(p_dst + oh * dst_w * c_blk_len, 0, dst_w * c_blk_len * sizeof(float));
                continue;
            }

            int64_t ow = 0;
            for (; ow < dst_kernel_start_w; ++ow) {
                maxpool2d_n16chw_border_fp32_sse(p_src, &param, oh, ow, p_dst);
            }
            for (; ow + POOLING_DST_W() <= dst_kernel_end_w; ow += POOLING_DST_W()) {
                maxpool2d_n16chw_1x4_kernel_func_table[stride_w_select][POOLING_DST_W()](p_src, &param, oh, ow, ihstart, ihend, p_dst);
            }
            if (ow < dst_kernel_end_w) {
                maxpool2d_n16chw_1x4_kernel_func_table[stride_w_select][dst_kernel_end_w - ow](p_src, &param, oh, ow, ihstart, ihend, p_dst);
                ow = dst_kernel_end_w;
            }
            for (; ow < dst_w; ++ow) {
                maxpool2d_n16chw_border_fp32_sse(p_src, &param, oh, ow, p_dst);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86