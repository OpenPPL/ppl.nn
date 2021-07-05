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

#include <nmmintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode relu_fp32_sse(
    const ppl::nn::TensorShape *x_shape,
    const float *x,
    float *y)
{
    const int64_t n_elem        = x_shape->GetElementsIncludingPadding();
    const int64_t simd_w        = 4;
    const int64_t unroll_n      = 4 * simd_w;
    const int64_t unroll_n_body = round(n_elem, unroll_n);

    if (unroll_n_body) {
        PRAGMA_OMP_PARALLEL()
        {
            __m128 mm_zero = _mm_setzero_ps();
            PRAGMA_OMP_FOR()
            for (int64_t n = 0; n < unroll_n_body; n += unroll_n) {
                _mm_storeu_ps(y + n + 0 * simd_w, _mm_max_ps(_mm_loadu_ps(x + n + 0 * simd_w), mm_zero));
                _mm_storeu_ps(y + n + 1 * simd_w, _mm_max_ps(_mm_loadu_ps(x + n + 1 * simd_w), mm_zero));
                _mm_storeu_ps(y + n + 2 * simd_w, _mm_max_ps(_mm_loadu_ps(x + n + 2 * simd_w), mm_zero));
                _mm_storeu_ps(y + n + 3 * simd_w, _mm_max_ps(_mm_loadu_ps(x + n + 3 * simd_w), mm_zero));
                // why the fucking compiler put the vzeroupper here?
            }
        }
    }
    for (int64_t n = unroll_n_body; n < n_elem; ++n) {
        y[n] = max(x[n], 0.0f);
    }
    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
