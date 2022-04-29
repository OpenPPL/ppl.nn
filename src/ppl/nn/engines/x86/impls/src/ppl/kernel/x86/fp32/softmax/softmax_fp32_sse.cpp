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

#include <math.h>
#include <float.h>
#include <nmmintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/math_sse.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode softmax_ndarray_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const int64_t axis,
    float *dst)
{
    const int64_t real_axis = axis < 0 ? axis + src_shape->GetDimCount() : axis;
    if (real_axis < 0 || real_axis >= src_shape->GetDimCount()) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int64_t outer_dim      = 1;
    int64_t inner_dim      = 1;
    for (int64_t i = 0; i < real_axis; i++) {
        outer_dim *= src_shape->GetDim(i);
    }
    for (int64_t i = real_axis; i < src_shape->GetDimCount(); i++) {
        inner_dim *= src_shape->GetDim(i);
    }

    const int64_t simd_w = 4;

PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < outer_dim; i++) {
        const float *p_src = src + i * inner_dim;
        float *p_dst       = dst + i * inner_dim;

        int64_t j;
        __m128 v_max_val = _mm_set1_ps(-FLT_MAX);
        float max_val = -FLT_MAX;
        float m_max_val[simd_w];
        for (j = 0; j + simd_w <= inner_dim; j += simd_w) {
            v_max_val = _mm_max_ps(v_max_val, _mm_loadu_ps(p_src + j));
        }
        _mm_storeu_ps(m_max_val, v_max_val);
        for (; j < inner_dim; j++) {
            max_val = max(max_val, p_src[j]);
        }
        for (j = 0; j < simd_w; j++) {
            max_val = max(max_val, m_max_val[j]);
        }
        v_max_val = _mm_set1_ps(max_val);

        float exp_sum    = 0;
        __m128 v_exp_sum = _mm_set1_ps(0);

        for (j = 0; j + simd_w <= inner_dim; j += simd_w) {
            const __m128 v_src     = _mm_loadu_ps(p_src + j);
            const __m128 v_exp_val = _sse_exp_ps(_mm_sub_ps(v_src, v_max_val));
            _mm_storeu_ps(p_dst + j, v_exp_val);
            v_exp_sum = _mm_add_ps(v_exp_sum, v_exp_val);
        }
        for (; j < inner_dim; j++) {
            float exp_val = expf(p_src[j] - max_val);
            p_dst[j]      = exp_val;
            exp_sum += exp_val;
        }

        if (inner_dim >= simd_w) {
            float temp[simd_w];
            _mm_storeu_ps(temp, v_exp_sum);
            for (int64_t k = 0; k < simd_w; k++) {
                exp_sum += temp[k];
            }
        }

        const float r_exp_sum    = 1.0f / exp_sum;
        const __m128 v_r_exp_sum = _mm_set1_ps(r_exp_sum);

        for (j = 0; j + simd_w <= inner_dim; j += simd_w) {
            __m128 v_dst = _mm_loadu_ps(p_dst + j);
            v_dst        = _mm_mul_ps(v_dst, v_r_exp_sum);
            _mm_storeu_ps(p_dst + j, v_dst);
        }
        for (; j < inner_dim; j++) {
            p_dst[j] *= r_exp_sum;
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode softmax13_ndarray_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const int64_t axis,
    float *dst)
{
    const int64_t real_axis = axis < 0 ? axis + src_shape->GetDimCount() : axis;
    if (real_axis < 0 || real_axis >= src_shape->GetDimCount()) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int64_t outer_dim      = 1;
    int64_t inner_dim      = 1;
    int64_t axis_dim       = src_shape->GetDim(real_axis);
    for (int64_t i = 0; i < real_axis; i++) {
        outer_dim *= src_shape->GetDim(i);
    }
    for (int64_t i = real_axis + 1; i < src_shape->GetDimCount(); i++) {
        inner_dim *= src_shape->GetDim(i);
    }

    if (inner_dim == 1)
        return softmax_ndarray_fp32_sse(src_shape, src, axis, dst);

    const int64_t simd_w = 4;

#ifndef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR()
#else
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
    for (int64_t i = 0; i < outer_dim; i++) {
        for (int64_t k = 0; k < inner_dim; k += simd_w) {
            const float *p_src = src + i * axis_dim * inner_dim + k;
            float *p_dst       = dst + i * axis_dim * inner_dim + k;
            const int64_t k_eff = min(inner_dim - k, simd_w);
            if (k_eff == simd_w) {
                // find max
                __m128 v_max_val = _mm_loadu_ps(p_src);
                for (int64_t j = 1; j < axis_dim; ++j) {
                    v_max_val = _mm_max_ps(_mm_loadu_ps(p_src + j * inner_dim), v_max_val);
                }

                __m128 v_exp_sum = _mm_setzero_ps();
                for (int64_t j = 0; j < axis_dim; ++j) {
                    const __m128 v_src     = _mm_loadu_ps(p_src + j * inner_dim);
                    const __m128 v_exp_val = _sse_exp_ps(_mm_sub_ps(v_src, v_max_val));
                    _mm_storeu_ps(p_dst + j * inner_dim, v_exp_val);
                    v_exp_sum = _mm_add_ps(v_exp_sum, v_exp_val);
                }

                const __m128 v_r_exp_sum = _mm_div_ps(_mm_set1_ps(1.0f), v_exp_sum);
                for (int64_t j = 0; j < axis_dim; ++j) {
                    __m128 v_dst = _mm_loadu_ps(p_dst + j * inner_dim);
                    v_dst        = _mm_mul_ps(v_dst, v_r_exp_sum);
                    _mm_storeu_ps(p_dst + j * inner_dim, v_dst);
                }
            } else {
                for (int64_t kk = 0; kk < k_eff; ++kk) {
                    const float *k_src = p_src + kk;
                    float *k_dst       = p_dst + kk;

                    // find max
                    float max_val = k_src[0];
                    for (int64_t j = 1; j < axis_dim; j++) {
                        if (k_src[j * inner_dim] > max_val) {
                            max_val = k_src[j * inner_dim];
                        }
                    }

                    float exp_sum = 0.0f;
                    for (int64_t j = 0; j < axis_dim; j++) {
                        float exp_val = expf(k_src[j * inner_dim] - max_val);
                        k_dst[j * inner_dim] = exp_val;
                        exp_sum += exp_val;
                    }

                    const float r_exp_sum = 1.0f / exp_sum;
                    for (int64_t j = 0; j < axis_dim; j++) {
                        k_dst[j * inner_dim] *= r_exp_sum;
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::kernel::x86
