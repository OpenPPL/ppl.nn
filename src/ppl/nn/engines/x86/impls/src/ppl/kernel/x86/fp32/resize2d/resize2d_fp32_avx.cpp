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

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode resize2d_n16chw_pytorch_2linear_floor_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const float scale_h,
    const float scale_w,
    float *dst)
{
    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t src_h    = src_shape->GetDim(2);
    const int64_t src_w    = src_shape->GetDim(3);
    const int64_t dst_h    = dst_shape->GetDim(2);
    const int64_t dst_w    = dst_shape->GetDim(3);
    const int64_t c_blk    = 16;
    const int64_t padded_c = round_up(channels, c_blk);
    const float hscale     = 1.0f / scale_h;
    const float wscale     = 1.0f / scale_w;
    const int64_t src_hw   = int64_t(src_h) * src_w;
    const int64_t dst_hw   = int64_t(dst_h) * dst_w;

    std::vector<int64_t> w0_vec(dst_w);
    std::vector<int64_t> w1_vec(dst_w);
    std::vector<float> w0_lambda_vec(dst_w);
    std::vector<float> w1_lambda_vec(dst_w);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t ow = 0; ow < dst_w; ow++) {
        float iw = dst_w > 1 ? (ow + 0.5f) * wscale - 0.5f : 0;
        if (iw < 0) {
            w0_vec[ow]        = 0;
            w1_vec[ow]        = 0;
            w0_lambda_vec[ow] = 1;
            w1_lambda_vec[ow] = 0;
        } else {
            w0_vec[ow]        = (int64_t)iw;
            w1_vec[ow]        = w0_vec[ow] + (w0_vec[ow] < src_w - 1);
            w1_lambda_vec[ow] = iw - w0_vec[ow];
            w0_lambda_vec[ow] = 1.0f - w1_lambda_vec[ow];
        }
    }

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t bc = 0; bc < batch * padded_c; bc += c_blk) {
        const float *l_src = src + bc * src_hw;
        float *l_dst       = dst + bc * dst_hw;
        for (int64_t oh = 0; oh < dst_h; ++oh) {
            float ih = dst_h > 1 ? (oh + 0.5f) * hscale - 0.5f : 0;
            int64_t h0, h1;
            float h0_lambda, h1_lambda;
            if (ih < 0) {
                h0        = 0;
                h1        = 0;
                h0_lambda = 1;
                h1_lambda = 0;
            } else {
                h0        = (int64_t)ih;
                h1        = h0 + (h0 < src_h - 1);
                h1_lambda = ih - h0;
                h0_lambda = 1.0f - h1_lambda;
            }

            for (int64_t ow = 0; ow < dst_w; ++ow) {
                int64_t w0, w1;
                float w0_lambda, w1_lambda;
                w0        = w0_vec[ow];
                w1        = w1_vec[ow];
                w0_lambda = w0_lambda_vec[ow];
                w1_lambda = w1_lambda_vec[ow];

                __m256 mm_w0 = _mm256_set1_ps(h0_lambda * w0_lambda);
                __m256 mm_w1 = _mm256_set1_ps(h0_lambda * w1_lambda);
                __m256 mm_w2 = _mm256_set1_ps(h1_lambda * w0_lambda);
                __m256 mm_w3 = _mm256_set1_ps(h1_lambda * w1_lambda);

                __m256 mm_dst0, mm_dst1;
                __m256 mm_i00, mm_i01, mm_i10, mm_i11;
                mm_dst0 = _mm256_mul_ps(_mm256_loadu_ps(l_src + (h0 * src_w + w0) * c_blk + 0), mm_w0);
                mm_dst1 = _mm256_mul_ps(_mm256_loadu_ps(l_src + (h0 * src_w + w0) * c_blk + 8), mm_w0);
                mm_i10  = _mm256_mul_ps(_mm256_loadu_ps(l_src + (h0 * src_w + w1) * c_blk + 0), mm_w1);
                mm_i11  = _mm256_mul_ps(_mm256_loadu_ps(l_src + (h0 * src_w + w1) * c_blk + 8), mm_w1);
                mm_dst0 = _mm256_add_ps(mm_i10, mm_dst0);
                mm_dst1 = _mm256_add_ps(mm_i11, mm_dst1);
                mm_i00  = _mm256_mul_ps(_mm256_loadu_ps(l_src + (h1 * src_w + w0) * c_blk + 0), mm_w2);
                mm_i01  = _mm256_mul_ps(_mm256_loadu_ps(l_src + (h1 * src_w + w0) * c_blk + 8), mm_w2);
                mm_dst0 = _mm256_add_ps(mm_i00, mm_dst0);
                mm_dst1 = _mm256_add_ps(mm_i01, mm_dst1);
                mm_i10  = _mm256_mul_ps(_mm256_loadu_ps(l_src + (h1 * src_w + w1) * c_blk + 0), mm_w3);
                mm_i11  = _mm256_mul_ps(_mm256_loadu_ps(l_src + (h1 * src_w + w1) * c_blk + 8), mm_w3);
                mm_dst0 = _mm256_add_ps(mm_i10, mm_dst0);
                mm_dst1 = _mm256_add_ps(mm_i11, mm_dst1);

                _mm256_storeu_ps(l_dst + (oh * dst_w + ow) * c_blk + 0, mm_dst0);
                _mm256_storeu_ps(l_dst + (oh * dst_w + ow) * c_blk + 8, mm_dst1);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode reisze2d_n16cx_asymmetric_nearest_floor_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const float scale_h,
    const float scale_w,
    float *dst)
{
    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t src_h    = src_shape->GetDim(2);
    const int64_t src_w    = src_shape->GetDim(3);
    const int64_t dst_h    = dst_shape->GetDim(2);
    const int64_t dst_w    = dst_shape->GetDim(3);
    const int64_t pad_c    = round_up(channels, 16);

    const float hscale = 1.0f / scale_h;
    const float wscale = 1.0f / scale_w;

    std::vector<int64_t> iw_list(dst_w);
    for (int64_t i = 0; i < dst_w; i++) {
        iw_list[i] = static_cast<int64_t>(i * wscale);
    }

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t ni = 0; ni < batch * pad_c; ni += 16) {
        const float *l_src = src + ni * src_h * src_w;
        float *l_dst       = dst + ni * dst_h * dst_w;
        for (int64_t oh = 0; oh < dst_h; ++oh) {
            int64_t ih         = static_cast<int64_t>(oh * hscale);
            float *t_dst       = l_dst + oh * dst_w * 16;
            const float *t_src = l_src + ih * src_w * 16;
            int64_t ow         = 0;
            for (; ow + 2 <= dst_w; ow += 2) {
                __m256 src0 = _mm256_loadu_ps(t_src + iw_list[ow] * 16 + 0);
                __m256 src1 = _mm256_loadu_ps(t_src + iw_list[ow] * 16 + 8);
                __m256 src2 = _mm256_loadu_ps(t_src + iw_list[ow + 1] * 16 + 0);
                __m256 src3 = _mm256_loadu_ps(t_src + iw_list[ow + 1] * 16 + 8);
                _mm256_storeu_ps(t_dst + (ow + 0) * 16 + 0, src0);
                _mm256_storeu_ps(t_dst + (ow + 0) * 16 + 8, src1);
                _mm256_storeu_ps(t_dst + (ow + 1) * 16 + 0, src2);
                _mm256_storeu_ps(t_dst + (ow + 1) * 16 + 8, src3);
            }
            for (; ow < dst_w; ow++) {
                __m256 src0 = _mm256_loadu_ps(t_src + iw_list[ow] * 16 + 0);
                __m256 src1 = _mm256_loadu_ps(t_src + iw_list[ow] * 16 + 8);
                _mm256_storeu_ps(t_dst + ow * 16 + 0, src0);
                _mm256_storeu_ps(t_dst + ow * 16 + 8, src1);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::kernel::x86
