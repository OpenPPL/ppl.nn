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
#include <immintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/math_avx512.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode softmax_ndarray_fp32_avx512(
    const ppl::common::TensorShape *src_shape,
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

    const int64_t simd_w = 16;

PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < outer_dim; i++) {
        const float *p_src = src + i * inner_dim;
        float *p_dst       = dst + i * inner_dim;

        int64_t j;
        __m512 v_max_val = _mm512_set1_ps(-FLT_MAX);
        float max_val = -FLT_MAX;
        float m_max_val[simd_w];
        for (j = 0; j + simd_w <= inner_dim; j += simd_w) {
            v_max_val = _mm512_max_ps(v_max_val, _mm512_loadu_ps(p_src + j));
        }
        _mm512_storeu_ps(m_max_val, v_max_val);
        for (; j < inner_dim; j++) {
            max_val = max(max_val, p_src[j]);
        }
        for (j = 0; j < simd_w; j++) {
            max_val = max(max_val, m_max_val[j]);
        }
        v_max_val = _mm512_set1_ps(max_val);

        float exp_sum    = 0;
        __m512 v_exp_sum = _mm512_set1_ps(0);
        for (j = 0; j + simd_w <= inner_dim; j += simd_w) {
            const __m512 v_src     = _mm512_loadu_ps(p_src + j);
            const __m512 v_exp_val = _avx512_exp_ps(_mm512_sub_ps(v_src, v_max_val));
            _mm512_storeu_ps(p_dst + j, v_exp_val);
            v_exp_sum = _mm512_add_ps(v_exp_sum, v_exp_val);
        }
        for (; j < inner_dim; j++) {
            float exp_val = expf(p_src[j] - max_val);
            p_dst[j]      = exp_val;
            exp_sum += exp_val;
        }

        if (inner_dim >= simd_w) {
            float temp[simd_w];
            _mm512_storeu_ps(temp, v_exp_sum);
            for (int64_t k = 0; k < simd_w; k++) {
                exp_sum += temp[k];
            }
        }

        const float r_exp_sum    = 1.0f / exp_sum;
        const __m512 v_r_exp_sum = _mm512_set1_ps(r_exp_sum);

        for (j = 0; j + simd_w <= inner_dim; j += simd_w) {
            __m512 v_dst = _mm512_loadu_ps(p_dst + j);
            v_dst        = _mm512_mul_ps(v_dst, v_r_exp_sum);
            _mm512_storeu_ps(p_dst + j, v_dst);
        }
        for (; j < inner_dim; j++) {
            p_dst[j] *= r_exp_sum;
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode softmax13_ndarray_fp32_avx512(
    const ppl::common::TensorShape *src_shape,
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
        return softmax_ndarray_fp32_avx512(src_shape, src, axis, dst);

    const int64_t simd_w = 16;

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
                __m512 v_max_val = _mm512_loadu_ps(p_src);
                for (int64_t j = 1; j < axis_dim; ++j) {
                    v_max_val = _mm512_max_ps(_mm512_loadu_ps(p_src + j * inner_dim), v_max_val);
                }

                __m512 v_exp_sum = _mm512_setzero_ps();
                for (int64_t j = 0; j < axis_dim; ++j) {
                    const __m512 v_src     = _mm512_loadu_ps(p_src + j * inner_dim);
                    const __m512 v_exp_val = _avx512_exp_ps(_mm512_sub_ps(v_src, v_max_val));
                    _mm512_storeu_ps(p_dst + j * inner_dim, v_exp_val);
                    v_exp_sum = _mm512_add_ps(v_exp_sum, v_exp_val);
                }

                const __m512 v_r_exp_sum = _mm512_div_ps(_mm512_set1_ps(1.0f), v_exp_sum);
                for (int64_t j = 0; j < axis_dim; ++j) {
                    __m512 v_dst = _mm512_loadu_ps(p_dst + j * inner_dim);
                    v_dst        = _mm512_mul_ps(v_dst, v_r_exp_sum);
                    _mm512_storeu_ps(p_dst + j * inner_dim, v_dst);
                }
            } else {
                __mmask16 mask = (1 << k_eff) - 1;
                __m512 v_max_val = _mm512_set1_ps(-FLT_MAX);
                for (int64_t j = 0; j < axis_dim; ++j) {
                    v_max_val = _mm512_max_ps(v_max_val, _mm512_maskz_load_ps(mask, p_src + j * inner_dim));
                }

                __m512 v_exp_sum = _mm512_setzero_ps();
                for (int64_t j = 0; j < axis_dim; ++j) {
                    const __m512 v_src     = _mm512_maskz_load_ps(mask, p_src + j * inner_dim);
                    const __m512 v_exp_val = _avx512_exp_ps(_mm512_sub_ps(v_src, v_max_val));
                    _mm512_mask_store_ps(p_dst + j * inner_dim, mask, v_exp_val);
                    v_exp_sum = _mm512_add_ps(v_exp_sum, v_exp_val);
                }

                const __m512 v_r_exp_sum = _mm512_div_ps(_mm512_set1_ps(1.0f), v_exp_sum);
                for (int64_t j = 0; j < axis_dim; ++j) {
                    __m512 v_dst = _mm512_maskz_load_ps(mask, p_dst + j * inner_dim);
                    v_dst        = _mm512_mul_ps(v_dst, v_r_exp_sum);
                    _mm512_mask_store_ps(p_dst + j * inner_dim, mask, v_dst);
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::kernel::x86
