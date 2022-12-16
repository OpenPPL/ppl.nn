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
#include <string.h> // for memset

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/averagepool2d/averagepool2d_common.h"

namespace ppl { namespace kernel { namespace x86 {

#define STRIDE_W_OPT()           3
#define POOLING_DST_W()          8
#define POOLING_CHANNELS_BLOCK() 16
#define SIMD_W()                 8

template <int64_t spec_stride_w, int64_t w_len>
static void averagepool2d_n16cx_1x8_kernel_fp32_avx(
    const float *src,
    const averagepool2d_param *param,
    const int64_t oh,
    const int64_t ow,
    const int64_t ihstart,
    const int64_t ihend,
    const int64_t pool_len,
    float *dst)
{
    const int64_t &kernel_w = param->kernel_w;
    const int64_t &pad_w    = param->pad_w;

    const int64_t &src_w = param->src_w;
    const int64_t &dst_w = param->dst_w;

    const int64_t stride_w = spec_stride_w ? spec_stride_w : param->stride_w;

    const int64_t iwstart = ow * stride_w - pad_w; // will always >= 0
    const int64_t iwend   = iwstart + kernel_w; // will always < src_w

    const int64_t c_blk_len = POOLING_CHANNELS_BLOCK();

    __m256 ymm00, ymm01, ymm02, ymm03;
    __m256 ymm04, ymm05, ymm06, ymm07;
    __m256 ymm08, ymm09, ymm10, ymm11;
    __m256 ymm12, ymm13, ymm14, ymm15;
    if (w_len >= 1) ymm00 = _mm256_setzero_ps();
    if (w_len >= 1) ymm01 = ymm00;
    if (w_len >= 2) ymm02 = ymm00;
    if (w_len >= 2) ymm03 = ymm00;
    if (w_len >= 3) ymm04 = ymm00;
    if (w_len >= 3) ymm05 = ymm00;
    if (w_len >= 4) ymm06 = ymm00;
    if (w_len >= 4) ymm07 = ymm00;
    if (w_len >= 5) ymm08 = ymm00;
    if (w_len >= 5) ymm09 = ymm00;
    if (w_len >= 6) ymm10 = ymm00;
    if (w_len >= 6) ymm11 = ymm00;
    if (w_len >= 7) ymm12 = ymm00;
    if (w_len >= 7) ymm13 = ymm00;
    if (w_len >= 8) ymm14 = ymm00;
    if (w_len >= 8) ymm15 = ymm00;

    for (int64_t ih = ihstart; ih < ihend; ++ih) {
        for (int64_t iw = iwstart; iw < iwend; ++iw) {
            const float *p_src = src + (ih * src_w + iw) * c_blk_len;
            if (w_len >= 1) ymm00 = _mm256_add_ps(ymm00, _mm256_loadu_ps(p_src + 0 * stride_w * c_blk_len + 0 * SIMD_W()));
            if (w_len >= 1) ymm01 = _mm256_add_ps(ymm01, _mm256_loadu_ps(p_src + 0 * stride_w * c_blk_len + 1 * SIMD_W()));
            if (w_len >= 2) ymm02 = _mm256_add_ps(ymm02, _mm256_loadu_ps(p_src + 1 * stride_w * c_blk_len + 0 * SIMD_W()));
            if (w_len >= 2) ymm03 = _mm256_add_ps(ymm03, _mm256_loadu_ps(p_src + 1 * stride_w * c_blk_len + 1 * SIMD_W()));
            if (w_len >= 3) ymm04 = _mm256_add_ps(ymm04, _mm256_loadu_ps(p_src + 2 * stride_w * c_blk_len + 0 * SIMD_W()));
            if (w_len >= 3) ymm05 = _mm256_add_ps(ymm05, _mm256_loadu_ps(p_src + 2 * stride_w * c_blk_len + 1 * SIMD_W()));
            if (w_len >= 4) ymm06 = _mm256_add_ps(ymm06, _mm256_loadu_ps(p_src + 3 * stride_w * c_blk_len + 0 * SIMD_W()));
            if (w_len >= 4) ymm07 = _mm256_add_ps(ymm07, _mm256_loadu_ps(p_src + 3 * stride_w * c_blk_len + 1 * SIMD_W()));
            if (w_len >= 5) ymm08 = _mm256_add_ps(ymm08, _mm256_loadu_ps(p_src + 4 * stride_w * c_blk_len + 0 * SIMD_W()));
            if (w_len >= 5) ymm09 = _mm256_add_ps(ymm09, _mm256_loadu_ps(p_src + 4 * stride_w * c_blk_len + 1 * SIMD_W()));
            if (w_len >= 6) ymm10 = _mm256_add_ps(ymm10, _mm256_loadu_ps(p_src + 5 * stride_w * c_blk_len + 0 * SIMD_W()));
            if (w_len >= 6) ymm11 = _mm256_add_ps(ymm11, _mm256_loadu_ps(p_src + 5 * stride_w * c_blk_len + 1 * SIMD_W()));
            if (w_len >= 7) ymm12 = _mm256_add_ps(ymm12, _mm256_loadu_ps(p_src + 6 * stride_w * c_blk_len + 0 * SIMD_W()));
            if (w_len >= 7) ymm13 = _mm256_add_ps(ymm13, _mm256_loadu_ps(p_src + 6 * stride_w * c_blk_len + 1 * SIMD_W()));
            if (w_len >= 8) ymm14 = _mm256_add_ps(ymm14, _mm256_loadu_ps(p_src + 7 * stride_w * c_blk_len + 0 * SIMD_W()));
            if (w_len >= 8) ymm15 = _mm256_add_ps(ymm15, _mm256_loadu_ps(p_src + 7 * stride_w * c_blk_len + 1 * SIMD_W()));
        }
    }

    float *p_dst        = dst + (oh * dst_w + ow) * c_blk_len;
    __m256 v_r_pool_len = _mm256_set1_ps(1.0f / pool_len);
    if (w_len >= 1) _mm256_storeu_ps(p_dst + 0 * c_blk_len + 0 * SIMD_W(), _mm256_mul_ps(ymm00, v_r_pool_len));
    if (w_len >= 1) _mm256_storeu_ps(p_dst + 0 * c_blk_len + 1 * SIMD_W(), _mm256_mul_ps(ymm01, v_r_pool_len));
    if (w_len >= 2) _mm256_storeu_ps(p_dst + 1 * c_blk_len + 0 * SIMD_W(), _mm256_mul_ps(ymm02, v_r_pool_len));
    if (w_len >= 2) _mm256_storeu_ps(p_dst + 1 * c_blk_len + 1 * SIMD_W(), _mm256_mul_ps(ymm03, v_r_pool_len));
    if (w_len >= 3) _mm256_storeu_ps(p_dst + 2 * c_blk_len + 0 * SIMD_W(), _mm256_mul_ps(ymm04, v_r_pool_len));
    if (w_len >= 3) _mm256_storeu_ps(p_dst + 2 * c_blk_len + 1 * SIMD_W(), _mm256_mul_ps(ymm05, v_r_pool_len));
    if (w_len >= 4) _mm256_storeu_ps(p_dst + 3 * c_blk_len + 0 * SIMD_W(), _mm256_mul_ps(ymm06, v_r_pool_len));
    if (w_len >= 4) _mm256_storeu_ps(p_dst + 3 * c_blk_len + 1 * SIMD_W(), _mm256_mul_ps(ymm07, v_r_pool_len));
    if (w_len >= 5) _mm256_storeu_ps(p_dst + 4 * c_blk_len + 0 * SIMD_W(), _mm256_mul_ps(ymm08, v_r_pool_len));
    if (w_len >= 5) _mm256_storeu_ps(p_dst + 4 * c_blk_len + 1 * SIMD_W(), _mm256_mul_ps(ymm09, v_r_pool_len));
    if (w_len >= 6) _mm256_storeu_ps(p_dst + 5 * c_blk_len + 0 * SIMD_W(), _mm256_mul_ps(ymm10, v_r_pool_len));
    if (w_len >= 6) _mm256_storeu_ps(p_dst + 5 * c_blk_len + 1 * SIMD_W(), _mm256_mul_ps(ymm11, v_r_pool_len));
    if (w_len >= 7) _mm256_storeu_ps(p_dst + 6 * c_blk_len + 0 * SIMD_W(), _mm256_mul_ps(ymm12, v_r_pool_len));
    if (w_len >= 7) _mm256_storeu_ps(p_dst + 6 * c_blk_len + 1 * SIMD_W(), _mm256_mul_ps(ymm13, v_r_pool_len));
    if (w_len >= 8) _mm256_storeu_ps(p_dst + 7 * c_blk_len + 0 * SIMD_W(), _mm256_mul_ps(ymm14, v_r_pool_len));
    if (w_len >= 8) _mm256_storeu_ps(p_dst + 7 * c_blk_len + 1 * SIMD_W(), _mm256_mul_ps(ymm15, v_r_pool_len));
}

typedef void (*averagepool2d_n16cx_kernel_fp32_avx_func_t)(const float *, const averagepool2d_param *, const int64_t, const int64_t, const int64_t, const int64_t, const int64_t, float *);
static const averagepool2d_n16cx_kernel_fp32_avx_func_t averagepool2d_n16cx_1x8_kernel_func_table[STRIDE_W_OPT()][POOLING_DST_W() + 1]{
    {
        averagepool2d_n16cx_1x8_kernel_fp32_avx<0, 0>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<0, 1>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<0, 2>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<0, 3>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<0, 4>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<0, 5>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<0, 6>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<0, 7>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<0, 8>,
    },
    {
        averagepool2d_n16cx_1x8_kernel_fp32_avx<1, 0>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<1, 1>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<1, 2>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<1, 3>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<1, 4>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<1, 5>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<1, 6>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<1, 7>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<1, 8>,
    },
    {
        averagepool2d_n16cx_1x8_kernel_fp32_avx<2, 0>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<2, 1>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<2, 2>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<2, 3>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<2, 4>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<2, 5>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<2, 6>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<2, 7>,
        averagepool2d_n16cx_1x8_kernel_fp32_avx<2, 8>,
    },
};

template <bool exclusive_mode, bool ceil_mode>
static inline void averagepool2d_n16cx_border_fp32_avx(
    const float *src,
    const averagepool2d_param *param,
    const int64_t oh,
    const int64_t ow,
    float *dst)
{
    const int64_t &kernel_h = param->kernel_h;
    const int64_t &kernel_w = param->kernel_w;
    const int64_t &stride_h = param->stride_h;
    const int64_t &stride_w = param->stride_w;
    const int64_t &pad_h    = param->pad_h;
    const int64_t &pad_w    = param->pad_w;

    const int64_t &src_h = param->src_h;
    const int64_t &src_w = param->src_w;
    const int64_t &dst_w = param->dst_w;

    const int64_t c_blk_len = POOLING_CHANNELS_BLOCK();

    const int64_t padded_ihstart = oh * stride_h - pad_h;
    const int64_t padded_iwstart = ow * stride_w - pad_w;
    const int64_t padded_ihend   = ceil_mode ? padded_ihstart + kernel_h : min<int64_t>(padded_ihstart + kernel_h, src_h + pad_h);
    const int64_t padded_iwend   = ceil_mode ? padded_iwstart + kernel_w : min<int64_t>(padded_iwstart + kernel_w, src_w + pad_w);

    const int64_t ihstart = max<int64_t>(padded_ihstart, 0);
    const int64_t iwstart = max<int64_t>(padded_iwstart, 0);
    const int64_t ihend   = min<int64_t>(padded_ihend, src_h);
    const int64_t iwend   = min<int64_t>(padded_iwend, src_w);

    int64_t pool_len = 0;
    if (exclusive_mode) {
        pool_len = (ihend - ihstart) * (iwend - iwstart);
    } else {
        pool_len = (padded_ihend - padded_ihstart) * (padded_iwend - padded_iwstart);
    }

    if (pool_len <= 0) {
        _mm256_storeu_ps(dst + (oh * dst_w + ow) * c_blk_len + 0 * SIMD_W(), _mm256_setzero_ps());
        _mm256_storeu_ps(dst + (oh * dst_w + ow) * c_blk_len + 1 * SIMD_W(), _mm256_setzero_ps());
    } else {
        __m256 v_r_pool_len = _mm256_set1_ps(1.0f / pool_len);
        __m256 v_sum_val0   = _mm256_setzero_ps();
        __m256 v_sum_val1   = v_sum_val0;
        for (int64_t ih = ihstart; ih < ihend; ++ih) {
            for (int64_t iw = iwstart; iw < iwend; ++iw) {
                __m256 v_src0 = _mm256_loadu_ps(src + (ih * src_w + iw) * c_blk_len + 0 * SIMD_W());
                __m256 v_src1 = _mm256_loadu_ps(src + (ih * src_w + iw) * c_blk_len + 1 * SIMD_W());
                v_sum_val0    = _mm256_add_ps(v_sum_val0, v_src0);
                v_sum_val1    = _mm256_add_ps(v_sum_val1, v_src1);
            }
        }
        _mm256_storeu_ps(dst + (oh * dst_w + ow) * c_blk_len + 0 * SIMD_W(), _mm256_mul_ps(v_sum_val0, v_r_pool_len));
        _mm256_storeu_ps(dst + (oh * dst_w + ow) * c_blk_len + 1 * SIMD_W(), _mm256_mul_ps(v_sum_val1, v_r_pool_len));
    }
}

template <bool exclusive_mode, bool ceil_mode>
ppl::common::RetCode averagepool2d_n16cx_blk1x8_fp32_avx_impl(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const float *src,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w,
    float *dst)
{
    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t src_h    = src_shape->GetDim(2);
    const int64_t src_w    = src_shape->GetDim(3);
    const int64_t dst_h    = dst_shape->GetDim(2);
    const int64_t dst_w    = dst_shape->GetDim(3);

    const averagepool2d_param param = {kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, batch, channels, src_h, src_w, dst_h, dst_w};

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
                    averagepool2d_n16cx_border_fp32_avx<exclusive_mode, ceil_mode>(p_src, &param, oh, ow, p_dst);
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

            const int64_t padded_ihstart = oh * stride_h - pad_h;
            const int64_t padded_ihend   = ceil_mode ? padded_ihstart + kernel_h : min<int64_t>(padded_ihstart + kernel_h, src_h + pad_h);
            const int64_t ihstart        = max<int64_t>(padded_ihstart, 0);
            const int64_t ihend          = min<int64_t>(padded_ihend, src_h);
            if (ihstart >= ihend) { // all input lines are padding lines
                memset(p_dst + oh * dst_w * c_blk_len, 0, dst_w * c_blk_len * sizeof(float));
                continue;
            }

            int64_t ow = 0;
            for (; ow < dst_kernel_start_w; ++ow) {
                averagepool2d_n16cx_border_fp32_avx<exclusive_mode, ceil_mode>(p_src, &param, oh, ow, p_dst);
            }
            int64_t kernel_pool_len = 0;
            if (exclusive_mode) {
                kernel_pool_len = (ihend - ihstart) * kernel_w;
            } else {
                kernel_pool_len = (padded_ihend - padded_ihstart) * kernel_w;
            }
            for (; ow + POOLING_DST_W() <= dst_kernel_end_w; ow += POOLING_DST_W()) {
                averagepool2d_n16cx_1x8_kernel_func_table[stride_w_select][POOLING_DST_W()](p_src, &param, oh, ow, ihstart, ihend, kernel_pool_len, p_dst);
            }
            if (ow < dst_kernel_end_w) {
                averagepool2d_n16cx_1x8_kernel_func_table[stride_w_select][dst_kernel_end_w - ow](p_src, &param, oh, ow, ihstart, ihend, kernel_pool_len, p_dst);
                ow = dst_kernel_end_w;
            }
            for (; ow < dst_w; ++ow) {
                averagepool2d_n16cx_border_fp32_avx<exclusive_mode, ceil_mode>(p_src, &param, oh, ow, p_dst);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode averagepool2d_n16cx_blk1x8_fp32_avx(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const float *src,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const bool exclusive_mode,
    const bool ceil_mode,
    float *dst)
{
    if (exclusive_mode) {
        if (ceil_mode) {
            return averagepool2d_n16cx_blk1x8_fp32_avx_impl<true, true>(src_shape, dst_shape, src, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dst);
        } else {
            return averagepool2d_n16cx_blk1x8_fp32_avx_impl<true, false>(src_shape, dst_shape, src, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dst);
        }
    } else {
        if (ceil_mode) {
            return averagepool2d_n16cx_blk1x8_fp32_avx_impl<false, true>(src_shape, dst_shape, src, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dst);
        } else {
            return averagepool2d_n16cx_blk1x8_fp32_avx_impl<false, false>(src_shape, dst_shape, src, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dst);
        }
    }

    return ppl::common::RC_INVALID_VALUE;
}

}}}; // namespace ppl::kernel::x86
