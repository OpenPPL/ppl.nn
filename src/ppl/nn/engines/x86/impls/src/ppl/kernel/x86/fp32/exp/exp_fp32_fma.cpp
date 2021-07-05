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

static inline __m256 _fma_exp_ps(__m256 x)
{
    __m256 tmp = _mm256_setzero_ps(), fx;
    __m256i imm0;
    __m256 one = _mm256_set1_ps(1.0f);

    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));

    fx = _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341), _mm256_set1_ps(0.5f));

    tmp = _mm256_floor_ps(fx);

    __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
    mask        = _mm256_and_ps(mask, one);
    fx          = _mm256_sub_ps(tmp, mask);

    tmp      = _mm256_mul_ps(fx, _mm256_set1_ps(0.693359375));
    __m256 z = _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4));
    x        = _mm256_sub_ps(x, tmp);
    x        = _mm256_sub_ps(x, z);
    z        = _mm256_mul_ps(x, x);

    __m256 y = _mm256_set1_ps(1.9875691500E-4);
    y        = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3));
    y        = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3));
    y        = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2));
    y        = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1));
    y        = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1));
    y        = _mm256_fmadd_ps(y, z, x);
    y        = _mm256_add_ps(y, one);

    imm0         = _mm256_cvttps_epi32(fx);
    imm0         = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
    imm0         = _mm256_slli_epi32(imm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(imm0);
    y            = _mm256_mul_ps(y, pow2n);
    return y;
}

ppl::common::RetCode exp_fp32_fma(
    const ppl::nn::TensorShape *x_shape,
    const float *x,
    float *y)
{
    const int64_t n_elem      = x_shape->GetElementsIncludingPadding();
    const int64_t unroll_n    = 32;
    const int64_t unroll_body = round(n_elem, unroll_n);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_n) {
        __m256 src0 = _mm256_loadu_ps(x + i + 0);
        __m256 src1 = _mm256_loadu_ps(x + i + 8);
        __m256 src2 = _mm256_loadu_ps(x + i + 16);
        __m256 src3 = _mm256_loadu_ps(x + i + 24);
        _mm256_storeu_ps(y + i + 0, _fma_exp_ps(src0));
        _mm256_storeu_ps(y + i + 8, _fma_exp_ps(src1));
        _mm256_storeu_ps(y + i + 16, _fma_exp_ps(src2));
        _mm256_storeu_ps(y + i + 24, _fma_exp_ps(src3));
    }
    for (int64_t i = unroll_body; i < n_elem; ++i) {
        y[i] = expf(x[i]);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86