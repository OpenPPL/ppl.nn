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

ppl::common::RetCode clip_fp32_avx(
    const ppl::nn::TensorShape *x_shape,
    const float *x,
    const float clip_min,
    const float clip_max,
    float *y)
{
    const int64_t n_elem        = x_shape->CalcElementsIncludingPadding();
    const int64_t simd_w        = 8;
    const int64_t unroll_n      = 4 * simd_w;
    const int64_t unroll_n_body = round(n_elem, unroll_n);

    if (unroll_n_body) {
        PRAGMA_OMP_PARALLEL()
        {
            __m256 mm_clip_min = _mm256_set1_ps(clip_min);
            __m256 mm_clip_max = _mm256_set1_ps(clip_max);
            PRAGMA_OMP_FOR()
            for (int64_t n = 0; n < unroll_n_body; n += unroll_n) {
                _mm256_storeu_ps(y + n + 0 * simd_w, _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(x + n + 0 * simd_w), mm_clip_min), mm_clip_max));
                _mm256_storeu_ps(y + n + 1 * simd_w, _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(x + n + 1 * simd_w), mm_clip_min), mm_clip_max));
                _mm256_storeu_ps(y + n + 2 * simd_w, _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(x + n + 2 * simd_w), mm_clip_min), mm_clip_max));
                _mm256_storeu_ps(y + n + 3 * simd_w, _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(x + n + 3 * simd_w), mm_clip_min), mm_clip_max));
            }
        }
    }
    for (int64_t n = unroll_n_body; n < n_elem; ++n) {
        y[n] = min(max(x[n], clip_min), clip_max);
    }
    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
