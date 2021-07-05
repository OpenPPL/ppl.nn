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

static inline __m256 _fma_sigmoid_ps(__m256 value)
{
    value = _mm256_max_ps(_mm256_set1_ps(-18.0f), value);
    value = _mm256_min_ps(_mm256_set1_ps(18.0f), value);

    __m256 value_squared = _mm256_mul_ps(value, value);

    __m256 p;
    p = _mm256_fmadd_ps(value_squared, _mm256_set1_ps(4.37031012579801e-11f), _mm256_set1_ps(1.15627324459942e-07f));
    p = _mm256_fmadd_ps(p, value_squared, _mm256_set1_ps(6.08574864600143e-05f));
    p = _mm256_fmadd_ps(p, value_squared, _mm256_set1_ps(8.51377133304701e-03f));
    p = _mm256_fmadd_ps(p, value_squared, _mm256_set1_ps(2.48287947061529e-01f));
    p = _mm256_mul_ps(p, value);

    __m256 q;
    q = _mm256_fmadd_ps(value_squared, _mm256_set1_ps(6.10247389755681e-13f), _mm256_set1_ps(5.76102136993427e-09f));
    q = _mm256_fmadd_ps(q, value_squared, _mm256_set1_ps(6.29106785017040e-06f));
    q = _mm256_fmadd_ps(q, value_squared, _mm256_set1_ps(1.70198817374094e-03f));
    q = _mm256_fmadd_ps(q, value_squared, _mm256_set1_ps(1.16817656904453e-01f));
    q = _mm256_fmadd_ps(q, value_squared, _mm256_set1_ps(9.93151921023180e-01f));

    __m256 dst = _mm256_add_ps(_mm256_div_ps(p, q), _mm256_set1_ps(0.5f));
    return dst;
}

ppl::common::RetCode sigmoid_fp32_fma(
    const ppl::nn::TensorShape *x_shape,
    const float *x,
    float *y)
{
    const int64_t n_elem      = x_shape->GetElementsIncludingPadding();
    const int64_t unroll_n    = 32;
    const int64_t unroll_body = round(n_elem, unroll_n);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_n) {
        __m256 src0, src1, src2, src3;
        src0 = _mm256_loadu_ps(x + i + 0);
        src1 = _mm256_loadu_ps(x + i + 8);
        src2 = _mm256_loadu_ps(x + i + 16);
        src3 = _mm256_loadu_ps(x + i + 24);
        _mm256_storeu_ps(y + i + 0, _fma_sigmoid_ps(src0));
        _mm256_storeu_ps(y + i + 8, _fma_sigmoid_ps(src1));
        _mm256_storeu_ps(y + i + 16, _fma_sigmoid_ps(src2));
        _mm256_storeu_ps(y + i + 24, _fma_sigmoid_ps(src3));
    }
    for (int64_t i = unroll_body; i < n_elem; ++i) {
        y[i] = 1.0f / (expf(-x[i]) + 1.0f);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
