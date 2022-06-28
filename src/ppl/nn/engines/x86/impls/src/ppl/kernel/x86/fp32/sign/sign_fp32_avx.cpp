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
#include "ppl/kernel/x86/common/math_avx.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode sign_fp32_avx(
    const ppl::nn::TensorShape *x_shape,
    const float *x,
    float *y)
{
    const int64_t n_elem      = x_shape->CalcElementsIncludingPadding();
    const int64_t unroll_n    = 16;
    const int64_t unroll_body = round(n_elem, unroll_n);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_n) {
        auto src0 = _mm256_loadu_ps(x + i + 0);
        auto src1 = _mm256_loadu_ps(x + i + 4);
        auto src2 = _mm256_loadu_ps(x + i + 8);
        auto src3 = _mm256_loadu_ps(x + i + 12);
        _mm256_storeu_ps(y + i + 0, _avx_sign_ps(src0));
        _mm256_storeu_ps(y + i + 4, _avx_sign_ps(src1));
        _mm256_storeu_ps(y + i + 8, _avx_sign_ps(src2));
        _mm256_storeu_ps(y + i + 12, _avx_sign_ps(src3));
    }
    for (int64_t i = unroll_body; i < n_elem; ++i) {
        y[i] = x[i] < 0.0f ? -1.0f : (x[i] == 0.0f ? 0.0f : 1.0f);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
