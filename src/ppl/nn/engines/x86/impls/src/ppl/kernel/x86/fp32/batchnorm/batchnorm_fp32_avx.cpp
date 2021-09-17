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
#include <math.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/avx_tools.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool fuse_relu>
ppl::common::RetCode batchnorm_ndarray_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const float *mean,
    const float *variance,
    const float *scale,
    const float *shift,
    const float var_eps,
    float *dst)
{
    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDimCount() > 1 ? src_shape->GetDim(1) : 1;
    int64_t inner_dim = 1;
    for (uint32_t i = 2; i < src_shape->GetDimCount(); ++i) {
        inner_dim *= src_shape->GetDim(i);
    }

    const int64_t simd_w = 8;
    const int64_t unroll_inner = 2 * simd_w;
    __m256 zero_vec      = _mm256_set1_ps(0.0f);
#ifndef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR()
#else
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
            const float mean_var    = mean[c];
            const float shift_var   = shift[c];
            const float var_rcp_var = scale[c] / sqrtf(variance[c] + var_eps);
            __m256 mean_vec0        = _mm256_set1_ps(mean_var);
            __m256 shift_vec0       = _mm256_set1_ps(shift_var);
            __m256 var_rcp_vec0     = _mm256_set1_ps(var_rcp_var);
            float *base_dst         = dst + n * channels * inner_dim + c * inner_dim;
            const float *base_src   = src + n * channels * inner_dim + c * inner_dim;
            for (int64_t i = 0; i < round(inner_dim, unroll_inner); i += unroll_inner) {
                __m256 data_vec0 = (_mm256_loadu_ps(base_src + simd_w * 0) - mean_vec0) * var_rcp_vec0 + shift_vec0;
                __m256 data_vec1 = (_mm256_loadu_ps(base_src + simd_w * 1) - mean_vec0) * var_rcp_vec0 + shift_vec0;
                if (fuse_relu) {
                    data_vec0 = _mm256_max_ps(data_vec0, zero_vec);
                    data_vec1 = _mm256_max_ps(data_vec1, zero_vec);
                }
                _mm256_storeu_ps(base_dst + simd_w * 0, data_vec0);
                _mm256_storeu_ps(base_dst + simd_w * 1, data_vec1);
                base_dst += unroll_inner;
                base_src += unroll_inner;
            }
            if (round(inner_dim, unroll_inner) < inner_dim) {
                for (int64_t i = 0; i < inner_dim - round(inner_dim, unroll_inner); ++i) {
                    float data = (base_src[i] - mean_var) * var_rcp_var + shift_var;
                    if (fuse_relu) {
                        data = max(data, 0.0f);
                    }
                    base_dst[i] = data;
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <bool fuse_relu>
ppl::common::RetCode batchnorm_n16cx_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const float *mean,
    const float *variance,
    const float *scale,
    const float *shift,
    const float var_eps,
    float *dst)
{
    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t height   = src_shape->GetDim(2);
    const int64_t width    = src_shape->GetDim(3);

    const int64_t simd_w = 8;
    const int64_t c_blk  = 16;
    const int64_t pad_c  = round_up(channels, c_blk);
    const int64_t hxw    = height * width;

    __m256 zero_vec = _mm256_set1_ps(0.0f);

    std::vector<float> temp_buffer_;
    if (channels < pad_c) {
        temp_buffer_.resize(c_blk * 4 * PPL_OMP_MAX_THREADS());
    }

#ifndef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR()
#else
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t c = 0; c < pad_c; c += c_blk) {
            const float *l_mean     = mean + c;
            const float *l_shift    = shift + c;
            const float *l_scale    = scale + c;
            const float *l_variance = variance + c;

            if (c + c_blk > channels) {
                float *temp_buffer            = temp_buffer_.data() + PPL_OMP_THREAD_ID() * c_blk * 4;
                const int32_t channels_remain = channels - round(channels, c_blk);
                for (int64_t i = 0; i < channels_remain; i++) {
                    temp_buffer[i + 0 * c_blk] = mean[i + c];
                    temp_buffer[i + 1 * c_blk] = shift[i + c];
                    temp_buffer[i + 2 * c_blk] = scale[i + c];
                    temp_buffer[i + 3 * c_blk] = variance[i + c];
                }
                l_mean     = temp_buffer + 0 * c_blk;
                l_shift    = temp_buffer + 1 * c_blk;
                l_scale    = temp_buffer + 2 * c_blk;
                l_variance = temp_buffer + 3 * c_blk;
            }

            const __m256 mean_vec0  = _mm256_loadu_ps(l_mean + simd_w * 0);
            const __m256 mean_vec1  = _mm256_loadu_ps(l_mean + simd_w * 1);
            const __m256 shift_vec0 = _mm256_loadu_ps(l_shift + simd_w * 0);
            const __m256 shift_vec1 = _mm256_loadu_ps(l_shift + simd_w * 1);

            // scale / sqrt(var + eps)
            const __m256 var_rcp_vec0 = _mm256_loadu_ps(l_scale + simd_w * 0) * _mm256_set1_ps(1.0f) / _mm256_sqrt_ps(_mm256_loadu_ps(l_variance + simd_w * 0) + _mm256_set1_ps(var_eps));
            const __m256 var_rcp_vec1 = _mm256_loadu_ps(l_scale + simd_w * 1) * _mm256_set1_ps(1.0f) / _mm256_sqrt_ps(_mm256_loadu_ps(l_variance + simd_w * 1) + _mm256_set1_ps(var_eps));

            float *base_dst       = dst + n * pad_c * hxw + c * hxw;
            const float *base_src = src + n * pad_c * hxw + c * hxw;

            for (int64_t i = 0; i < hxw; i++) {
                __m256 data_vec0 = (_mm256_loadu_ps(base_src + simd_w * 0) - mean_vec0) * var_rcp_vec0 + shift_vec0;
                __m256 data_vec1 = (_mm256_loadu_ps(base_src + simd_w * 1) - mean_vec1) * var_rcp_vec1 + shift_vec1;
                if (fuse_relu) {
                    data_vec0 = _mm256_max_ps(data_vec0, zero_vec);
                    data_vec1 = _mm256_max_ps(data_vec1, zero_vec);
                }
                _mm256_storeu_ps(base_dst + simd_w * 0, data_vec0);
                _mm256_storeu_ps(base_dst + simd_w * 1, data_vec1);
                base_dst += c_blk;
                base_src += c_blk;
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode batchnorm_ndarray_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const float *mean,
    const float *variance,
    const float *scale,
    const float *shift,
    const float var_eps,
    const bool relu,
    float *dst)
{
    if (relu) {
        return batchnorm_ndarray_fp32_avx<true>(src_shape, src, mean, variance, scale, shift, var_eps, dst);
    } else {
        return batchnorm_ndarray_fp32_avx<false>(src_shape, src, mean, variance, scale, shift, var_eps, dst);
    }
}

ppl::common::RetCode batchnorm_n16cx_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const float *mean,
    const float *variance,
    const float *scale,
    const float *shift,
    const float var_eps,
    const bool relu,
    float *dst)
{
    if (relu) {
        return batchnorm_n16cx_fp32_avx<true>(src_shape, src, mean, variance, scale, shift, var_eps, dst);
    } else {
        return batchnorm_n16cx_fp32_avx<false>(src_shape, src, mean, variance, scale, shift, var_eps, dst);
    }
}

}}}; // namespace ppl::kernel::x86