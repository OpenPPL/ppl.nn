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
#include "ppl/kernel/x86/common/math_sse.h"

namespace ppl { namespace kernel { namespace x86 {

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
