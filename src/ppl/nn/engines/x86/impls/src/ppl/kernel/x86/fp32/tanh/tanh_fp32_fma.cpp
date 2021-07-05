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

namespace ppl { namespace kernel { namespace x86 {

// an approximation of tanh
static inline __m256 _fma_tanh_ps(__m256 value)
{
    value = _mm256_max_ps(_mm256_set1_ps(-9.0f), value);
    value = _mm256_min_ps(_mm256_set1_ps(9.0f), value);

    __m256 value_squared = _mm256_mul_ps(value, value);

    __m256 p;
    p = _mm256_fmadd_ps(value_squared, _mm256_set1_ps(-2.76076847742355e-16f), _mm256_set1_ps(2.00018790482477e-13f));
    p = _mm256_fmadd_ps(p, value_squared, _mm256_set1_ps(-8.60467152213735e-11f));
    p = _mm256_fmadd_ps(p, value_squared, _mm256_set1_ps(5.12229709037114e-08f));
    p = _mm256_fmadd_ps(p, value_squared, _mm256_set1_ps(1.48572235717979e-05f));
    p = _mm256_fmadd_ps(p, value_squared, _mm256_set1_ps(6.37261928875436e-04f));
    p = _mm256_fmadd_ps(p, value_squared, _mm256_set1_ps(4.89352455891786e-03f));
    p = _mm256_mul_ps(p, value);

    __m256 q;
    q = _mm256_fmadd_ps(value_squared, _mm256_set1_ps(1.19825839466702e-06f), _mm256_set1_ps(1.18534705686654e-04f));
    q = _mm256_fmadd_ps(q, value_squared, _mm256_set1_ps(2.26843463243900e-03f));
    q = _mm256_fmadd_ps(q, value_squared, _mm256_set1_ps(4.89352518554385e-03f));

    __m256 dst = _mm256_div_ps(p, q);
    return dst;
}

ppl::common::RetCode tanh_fp32_fma(
    const ppl::nn::TensorShape *x_shape,
    const float *x,
    float *y)
{
    const int64_t simd_w      = 8;
    const int64_t n_elem      = x_shape->GetElementsIncludingPadding();
    const int64_t unroll_n    = 4 * simd_w;
    const int64_t unroll_body = round(n_elem, unroll_n);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_n) {
        __m256 v_src0 = _mm256_loadu_ps(x + i + 0 * simd_w);
        __m256 v_src1 = _mm256_loadu_ps(x + i + 1 * simd_w);
        __m256 v_src2 = _mm256_loadu_ps(x + i + 2 * simd_w);
        __m256 v_src3 = _mm256_loadu_ps(x + i + 3 * simd_w);
        _mm256_storeu_ps(y + i + 0 * simd_w, _fma_tanh_ps(v_src0));
        _mm256_storeu_ps(y + i + 1 * simd_w, _fma_tanh_ps(v_src1));
        _mm256_storeu_ps(y + i + 2 * simd_w, _fma_tanh_ps(v_src2));
        _mm256_storeu_ps(y + i + 3 * simd_w, _fma_tanh_ps(v_src3));
    }
    for (int64_t i = unroll_body; i < n_elem; ++i) {
        y[i] = tanh(x[i]);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
