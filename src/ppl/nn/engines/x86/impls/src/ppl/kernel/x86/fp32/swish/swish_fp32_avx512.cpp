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
#include "ppl/kernel/x86/common/math_avx512.h"
#include "ppl/kernel/x86/common/avx512_tools.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode swish_fp32_avx512(
    const ppl::nn::TensorShape *x_shape,
    const float *x,
    const float beta,
    float *y)
{
    const int64_t V_REG_ELTS = 16;

    const int64_t n_elem      = x_shape->CalcElementsIncludingPadding();
    const int64_t unroll_n    = 4 * V_REG_ELTS;
    const int64_t unroll_body = round(n_elem, unroll_n);

    const auto v_beta = _mm512_set1_ps(beta);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_n) {
        auto src0 = _mm512_loadu_ps(x + i + 0 * V_REG_ELTS);
        auto src1 = _mm512_loadu_ps(x + i + 1 * V_REG_ELTS);
        auto src2 = _mm512_loadu_ps(x + i + 2 * V_REG_ELTS);
        auto src3 = _mm512_loadu_ps(x + i + 3 * V_REG_ELTS);
        _mm512_storeu_ps(y + i + 0 * V_REG_ELTS, src0 * _avx512_sigmoid_ps(v_beta * src0));
        _mm512_storeu_ps(y + i + 1 * V_REG_ELTS, src1 * _avx512_sigmoid_ps(v_beta * src1));
        _mm512_storeu_ps(y + i + 2 * V_REG_ELTS, src2 * _avx512_sigmoid_ps(v_beta * src2));
        _mm512_storeu_ps(y + i + 3 * V_REG_ELTS, src3 * _avx512_sigmoid_ps(v_beta * src3));
    }
    int64_t i = unroll_body;
    for (; i <= n_elem - V_REG_ELTS; i += V_REG_ELTS) {
        auto src0 = _mm512_loadu_ps(x + i + 0 * V_REG_ELTS);
        _mm512_storeu_ps(y + i + 0 * V_REG_ELTS, src0 * _avx512_sigmoid_ps(v_beta * src0));
    }
    if (i < n_elem) {
        __mmask16 mask = static_cast<__mmask16>((1 << (n_elem - i)) - 1);
        auto src0 = _mm512_maskz_loadu_ps(mask, x + i + 0 * V_REG_ELTS);
        _mm512_mask_storeu_ps(y + i + 0 * V_REG_ELTS, mask, src0 * _avx512_sigmoid_ps(v_beta * src0));
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
