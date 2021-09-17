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
#include "ppl/kernel/x86/common/math_fma.h"
#include "ppl/kernel/x86/common/avx_tools.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode swish_fp32_fma(
    const ppl::nn::TensorShape *x_shape,
    const float *x,
    const float beta,
    float *y)
{
    const __m256 v_beta       = _mm256_set1_ps(beta);
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
        _mm256_storeu_ps(y + i + 0, src0 * _fma_sigmoid_ps(v_beta * src0));
        _mm256_storeu_ps(y + i + 8, src1 * _fma_sigmoid_ps(v_beta * src1));
        _mm256_storeu_ps(y + i + 16, src2 * _fma_sigmoid_ps(v_beta * src2));
        _mm256_storeu_ps(y + i + 24, src3 * _fma_sigmoid_ps(v_beta * src3));
    }
    for (int64_t i = unroll_body; i < n_elem; ++i) {
        const float src_val = x[i];
        y[i]                = src_val / (expf(-beta * src_val) + 1.0f);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
