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
#include "ppl/nn/params/onnx/pooling_param.h"

namespace ppl { namespace kernel { namespace x86 {

#define STRIDE_W_OPT()           3
#define POOLING_DST_W()          16
#define POOLING_CHANNELS_BLOCK() 16

template <int64_t spec_stride_w, int64_t w_len>
static void averagepool2d_n16chw_1x16_kernel_fp32_avx512(
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

    __m512 zmm00, zmm01, zmm02, zmm03;
    __m512 zmm04, zmm05, zmm06, zmm07;
    __m512 zmm08, zmm09, zmm10, zmm11;
    __m512 zmm12, zmm13, zmm14, zmm15;
    if (w_len >= 1) zmm00 = _mm512_setzero_ps();
    if (w_len >= 2) zmm01 = zmm00;
    if (w_len >= 3) zmm02 = zmm00;
    if (w_len >= 4) zmm03 = zmm00;
    if (w_len >= 5) zmm04 = zmm00;
    if (w_len >= 6) zmm05 = zmm00;
    if (w_len >= 7) zmm06 = zmm00;
    if (w_len >= 8) zmm07 = zmm00;
    if (w_len >= 9) zmm08 = zmm00;
    if (w_len >= 10) zmm09 = zmm00;
    if (w_len >= 11) zmm10 = zmm00;
    if (w_len >= 12) zmm11 = zmm00;
    if (w_len >= 13) zmm12 = zmm00;
    if (w_len >= 14) zmm13 = zmm00;
    if (w_len >= 15) zmm14 = zmm00;
    if (w_len >= 16) zmm15 = zmm00;

    for (int64_t ih = ihstart; ih < ihend; ++ih) {
        for (int64_t iw = iwstart; iw < iwend; ++iw) {
            const float *p_src = src + (ih * src_w + iw) * c_blk_len;
            if (w_len >= 1) zmm00 = _mm512_add_ps(zmm00, _mm512_loadu_ps(p_src + 0 * stride_w * c_blk_len));
            if (w_len >= 2) zmm01 = _mm512_add_ps(zmm01, _mm512_loadu_ps(p_src + 1 * stride_w * c_blk_len));
            if (w_len >= 3) zmm02 = _mm512_add_ps(zmm02, _mm512_loadu_ps(p_src + 2 * stride_w * c_blk_len));
            if (w_len >= 4) zmm03 = _mm512_add_ps(zmm03, _mm512_loadu_ps(p_src + 3 * stride_w * c_blk_len));
            if (w_len >= 5) zmm04 = _mm512_add_ps(zmm04, _mm512_loadu_ps(p_src + 4 * stride_w * c_blk_len));
            if (w_len >= 6) zmm05 = _mm512_add_ps(zmm05, _mm512_loadu_ps(p_src + 5 * stride_w * c_blk_len));
            if (w_len >= 7) zmm06 = _mm512_add_ps(zmm06, _mm512_loadu_ps(p_src + 6 * stride_w * c_blk_len));
            if (w_len >= 8) zmm07 = _mm512_add_ps(zmm07, _mm512_loadu_ps(p_src + 7 * stride_w * c_blk_len));
            if (w_len >= 9) zmm08 = _mm512_add_ps(zmm08, _mm512_loadu_ps(p_src + 8 * stride_w * c_blk_len));
            if (w_len >= 10) zmm09 = _mm512_add_ps(zmm09, _mm512_loadu_ps(p_src + 9 * stride_w * c_blk_len));
            if (w_len >= 11) zmm10 = _mm512_add_ps(zmm10, _mm512_loadu_ps(p_src + 10 * stride_w * c_blk_len));
            if (w_len >= 12) zmm11 = _mm512_add_ps(zmm11, _mm512_loadu_ps(p_src + 11 * stride_w * c_blk_len));
            if (w_len >= 13) zmm12 = _mm512_add_ps(zmm12, _mm512_loadu_ps(p_src + 12 * stride_w * c_blk_len));
            if (w_len >= 14) zmm13 = _mm512_add_ps(zmm13, _mm512_loadu_ps(p_src + 13 * stride_w * c_blk_len));
            if (w_len >= 15) zmm14 = _mm512_add_ps(zmm14, _mm512_loadu_ps(p_src + 14 * stride_w * c_blk_len));
            if (w_len >= 16) zmm15 = _mm512_add_ps(zmm15, _mm512_loadu_ps(p_src + 15 * stride_w * c_blk_len));
        }
    }

    float *p_dst        = dst + (oh * dst_w + ow) * c_blk_len;
    __m512 v_r_pool_len = _mm512_set1_ps(1.0f / pool_len);
    if (w_len >= 1) _mm512_storeu_ps(p_dst + 0 * c_blk_len, _mm512_mul_ps(zmm00, v_r_pool_len));
    if (w_len >= 2) _mm512_storeu_ps(p_dst + 1 * c_blk_len, _mm512_mul_ps(zmm01, v_r_pool_len));
    if (w_len >= 3) _mm512_storeu_ps(p_dst + 2 * c_blk_len, _mm512_mul_ps(zmm02, v_r_pool_len));
    if (w_len >= 4) _mm512_storeu_ps(p_dst + 3 * c_blk_len, _mm512_mul_ps(zmm03, v_r_pool_len));
    if (w_len >= 5) _mm512_storeu_ps(p_dst + 4 * c_blk_len, _mm512_mul_ps(zmm04, v_r_pool_len));
    if (w_len >= 6) _mm512_storeu_ps(p_dst + 5 * c_blk_len, _mm512_mul_ps(zmm05, v_r_pool_len));
    if (w_len >= 7) _mm512_storeu_ps(p_dst + 6 * c_blk_len, _mm512_mul_ps(zmm06, v_r_pool_len));
    if (w_len >= 8) _mm512_storeu_ps(p_dst + 7 * c_blk_len, _mm512_mul_ps(zmm07, v_r_pool_len));
    if (w_len >= 9) _mm512_storeu_ps(p_dst + 8 * c_blk_len, _mm512_mul_ps(zmm08, v_r_pool_len));
    if (w_len >= 10) _mm512_storeu_ps(p_dst + 9 * c_blk_len, _mm512_mul_ps(zmm09, v_r_pool_len));
    if (w_len >= 11) _mm512_storeu_ps(p_dst + 10 * c_blk_len, _mm512_mul_ps(zmm10, v_r_pool_len));
    if (w_len >= 12) _mm512_storeu_ps(p_dst + 11 * c_blk_len, _mm512_mul_ps(zmm11, v_r_pool_len));
    if (w_len >= 13) _mm512_storeu_ps(p_dst + 12 * c_blk_len, _mm512_mul_ps(zmm12, v_r_pool_len));
    if (w_len >= 14) _mm512_storeu_ps(p_dst + 13 * c_blk_len, _mm512_mul_ps(zmm13, v_r_pool_len));
    if (w_len >= 15) _mm512_storeu_ps(p_dst + 14 * c_blk_len, _mm512_mul_ps(zmm14, v_r_pool_len));
    if (w_len >= 16) _mm512_storeu_ps(p_dst + 15 * c_blk_len, _mm512_mul_ps(zmm15, v_r_pool_len));
}

typedef void (*averagepool2d_n16chw_kernel_fp32_avx512_func_t)(const float *, const averagepool2d_param *, const int64_t, const int64_t, const int64_t, const int64_t, const int64_t, float *);
static const averagepool2d_n16chw_kernel_fp32_avx512_func_t averagepool2d_n16chw_1x16_kernel_func_table[STRIDE_W_OPT()][POOLING_DST_W() + 1]{
    {
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<0, 0>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<0, 1>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<0, 2>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<0, 3>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<0, 4>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<0, 5>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<0, 6>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<0, 7>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<0, 8>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<0, 9>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<0, 10>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<0, 11>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<0, 12>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<0, 13>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<0, 14>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<0, 15>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<0, 16>,
    },
    {
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<1, 0>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<1, 1>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<1, 2>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<1, 3>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<1, 4>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<1, 5>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<1, 6>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<1, 7>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<1, 8>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<1, 9>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<1, 10>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<1, 11>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<1, 12>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<1, 13>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<1, 14>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<1, 15>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<1, 16>,
    },
    {
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<2, 0>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<2, 1>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<2, 2>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<2, 3>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<2, 4>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<2, 5>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<2, 6>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<2, 7>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<2, 8>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<2, 9>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<2, 10>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<2, 11>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<2, 12>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<2, 13>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<2, 14>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<2, 15>,
        averagepool2d_n16chw_1x16_kernel_fp32_avx512<2, 16>,
    },
};

template <ppl::nn::onnx::PoolingParam::pooling_mode_t pooling_mode, bool ceil_mode>
static inline void averagepool2d_n16chw_border_fp32_avx512(
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
    if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE) {
        pool_len = (ihend - ihstart) * (iwend - iwstart);
    } else if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE) {
        pool_len = (padded_ihend - padded_ihstart) * (padded_iwend - padded_iwstart);
    }

    if (pool_len <= 0) {
        _mm512_storeu_ps(dst + (oh * dst_w + ow) * c_blk_len, _mm512_setzero_ps());
    } else {
        __m512 v_r_pool_len = _mm512_set1_ps(1.0f / pool_len);
        __m512 v_sum_val    = _mm512_setzero_ps();
        for (int64_t ih = ihstart; ih < ihend; ++ih) {
            for (int64_t iw = iwstart; iw < iwend; ++iw) {
                __m512 v_src = _mm512_loadu_ps(src + (ih * src_w + iw) * c_blk_len);
                v_sum_val    = _mm512_add_ps(v_sum_val, v_src);
            }
        }
        _mm512_storeu_ps(dst + (oh * dst_w + ow) * c_blk_len, _mm512_mul_ps(v_sum_val, v_r_pool_len));
    }
}

template <ppl::nn::onnx::PoolingParam::pooling_mode_t pooling_mode, bool ceil_mode>
ppl::common::RetCode averagepool2d_n16chw_blk1x16_fp32_avx512_impl(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
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
                    averagepool2d_n16chw_border_fp32_avx512<pooling_mode, ceil_mode>(p_src, &param, oh, ow, p_dst);
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
                averagepool2d_n16chw_border_fp32_avx512<pooling_mode, ceil_mode>(p_src, &param, oh, ow, p_dst);
            }
            int64_t kernel_pool_len = 0;
            if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE) {
                kernel_pool_len = (ihend - ihstart) * kernel_w;
            } else if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE) {
                kernel_pool_len = (padded_ihend - padded_ihstart) * kernel_w;
            }
            for (; ow + POOLING_DST_W() <= dst_kernel_end_w; ow += POOLING_DST_W()) {
                averagepool2d_n16chw_1x16_kernel_func_table[stride_w_select][POOLING_DST_W()](p_src, &param, oh, ow, ihstart, ihend, kernel_pool_len, p_dst);
            }
            if (ow < dst_kernel_end_w) {
                averagepool2d_n16chw_1x16_kernel_func_table[stride_w_select][dst_kernel_end_w - ow](p_src, &param, oh, ow, ihstart, ihend, kernel_pool_len, p_dst);
                ow = dst_kernel_end_w;
            }
            for (; ow < dst_w; ++ow) {
                averagepool2d_n16chw_border_fp32_avx512<pooling_mode, ceil_mode>(p_src, &param, oh, ow, p_dst);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode averagepool2d_n16chw_blk1x16_fp32_avx512(
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
    float *dst)
{
    if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE) {
        if (ceil_mode) {
            return averagepool2d_n16chw_blk1x16_fp32_avx512_impl<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, true>(src_shape, dst_shape, src, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dst);
        } else {
            return averagepool2d_n16chw_blk1x16_fp32_avx512_impl<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, false>(src_shape, dst_shape, src, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dst);
        }
    } else if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE) {
        if (ceil_mode) {
            return averagepool2d_n16chw_blk1x16_fp32_avx512_impl<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, true>(src_shape, dst_shape, src, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dst);
        } else {
            return averagepool2d_n16chw_blk1x16_fp32_avx512_impl<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, false>(src_shape, dst_shape, src, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dst);
        }
    }

    return ppl::common::RC_INVALID_VALUE;
}

}}}; // namespace ppl::kernel::x86