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
#include <nmmintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

// an approximation of exp
static inline __m128 _sse_exp_ps(__m128 x)
{
    __m128 tmp = _mm_setzero_ps(), fx;
    __m128i imm0;
    __m128 one = _mm_set1_ps(1.0f);

    x = _mm_min_ps(x, _mm_set1_ps(88.3762626647949f));
    x = _mm_max_ps(x, _mm_set1_ps(-88.3762626647949f));

    fx = _mm_mul_ps(x, _mm_set1_ps(1.44269504088896341));
    fx = _mm_add_ps(fx, _mm_set1_ps(0.5f));

    tmp = _mm_floor_ps(fx);

    __m128 mask = _mm_cmpgt_ps(tmp, fx);
    mask        = _mm_and_ps(mask, one);
    fx          = _mm_sub_ps(tmp, mask);

    tmp      = _mm_mul_ps(fx, _mm_set1_ps(0.693359375));
    __m128 z = _mm_mul_ps(fx, _mm_set1_ps(-2.12194440e-4));
    x        = _mm_sub_ps(x, tmp);
    x        = _mm_sub_ps(x, z);
    z        = _mm_mul_ps(x, x);

    __m128 y = _mm_set1_ps(1.9875691500E-4);
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _mm_set1_ps(1.3981999507E-3));
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _mm_set1_ps(8.3334519073E-3));
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _mm_set1_ps(4.1665795894E-2));
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _mm_set1_ps(1.6666665459E-1));
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _mm_set1_ps(5.0000001201E-1));
    y        = _mm_mul_ps(y, z);
    y        = _mm_add_ps(y, x);
    y        = _mm_add_ps(y, one);

    imm0         = _mm_cvttps_epi32(fx);
    imm0         = _mm_add_epi32(imm0, _mm_set1_epi32(0x7f));
    imm0         = _mm_slli_epi32(imm0, 23);
    __m128 pow2n = _mm_castsi128_ps(imm0);
    y            = _mm_mul_ps(y, pow2n);
    return y;
}

ppl::common::RetCode softmax_ndarray_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const int64_t axis,
    float *dst)
{
    int64_t outer_dim      = 1;
    int64_t inner_dim      = 1;
    const int64_t axis_dim = src_shape->GetDim(axis);
    for (int64_t i = 0; i < axis; i++) {
        outer_dim *= src_shape->GetDim(i);
    }
    for (int64_t i = axis + 1; i < src_shape->GetDimCount(); i++) {
        inner_dim *= src_shape->GetDim(i);
    }

    const int64_t simd_w = 4;

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < outer_dim; i++) {
        const float *p_src = src + i * axis_dim * inner_dim;
        float *p_dst       = dst + i * axis_dim * inner_dim;

        float exp_sum    = 0;
        __m128 v_exp_sum = _mm_set1_ps(0);
        int64_t j        = 0;
        for (; j + simd_w <= axis_dim * inner_dim; j += simd_w) {
            const __m128 v_src     = _mm_loadu_ps(p_src + j);
            const __m128 v_exp_val = _sse_exp_ps(v_src);
            _mm_storeu_ps(p_dst + j, v_exp_val);
            v_exp_sum = _mm_add_ps(v_exp_sum, v_exp_val);
        }
        for (; j < axis_dim * inner_dim; j++) {
            float exp_val = expf(p_src[j]);
            p_dst[j]      = exp_val;
            exp_sum += exp_val;
        }

        if (axis_dim * inner_dim >= simd_w) {
            float temp[simd_w];
            _mm_storeu_ps(temp, v_exp_sum);
            for (int64_t k = 0; k < simd_w; k++) {
                exp_sum += temp[k];
            }
        }

        const float r_exp_sum    = 1.0f / exp_sum;
        const __m128 v_r_exp_sum = _mm_set1_ps(r_exp_sum);

        j = 0;
        for (; j + simd_w <= axis_dim * inner_dim; j += simd_w) {
            __m128 v_dst = _mm_loadu_ps(p_dst + j);
            v_dst        = _mm_mul_ps(v_dst, v_r_exp_sum);
            _mm_storeu_ps(p_dst + j, v_dst);
        }
        for (; j < axis_dim * inner_dim; j++) {
            p_dst[j] *= r_exp_sum;
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::kernel::x86
