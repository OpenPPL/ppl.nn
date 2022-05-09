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

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/abs.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode abs_fp32_ref(
    const ppl::nn::TensorShape *x_shape,
    const float *x,
    float *y)
{
#define _OP_SS(Y, X) \
    do {             \
        Y = fabsf(X); \
    } while (0)

    const int64_t n_elem      = x_shape->GetElementsIncludingPadding();
    const int64_t unroll_n    = 16;
    const int64_t unroll_body = round(n_elem, unroll_n);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_n) {
        _OP_SS(y[i + 0], x[i + 0]);
        _OP_SS(y[i + 8 + 0], x[i + 8 + 0]);
        _OP_SS(y[i + 1], x[i + 1]);
        _OP_SS(y[i + 8 + 1], x[i + 8 + 1]);
        _OP_SS(y[i + 2], x[i + 2]);
        _OP_SS(y[i + 8 + 2], x[i + 8 + 2]);
        _OP_SS(y[i + 3], x[i + 3]);
        _OP_SS(y[i + 8 + 3], x[i + 8 + 3]);
        _OP_SS(y[i + 4], x[i + 4]);
        _OP_SS(y[i + 8 + 4], x[i + 8 + 4]);
        _OP_SS(y[i + 5], x[i + 5]);
        _OP_SS(y[i + 8 + 5], x[i + 8 + 5]);
        _OP_SS(y[i + 6], x[i + 6]);
        _OP_SS(y[i + 8 + 6], x[i + 8 + 6]);
        _OP_SS(y[i + 7], x[i + 7]);
        _OP_SS(y[i + 8 + 7], x[i + 8 + 7]);
    }
    for (int64_t i = unroll_body; i < n_elem; ++i) {
        _OP_SS(y[i + 0], x[i + 0]);
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode abs_fp32(
    const ppl::common::isa_t isa,
    const ppl::nn::TensorShape *x_shape,
    const float *x,
    float *y)
{
    if (isa & ppl::common::ISA_X86_AVX) {
        return abs_fp32_avx(x_shape, x, y);
    }
    if (isa & ppl::common::ISA_X86_SSE) {
        return abs_fp32_sse(x_shape, x, y);
    }
    return abs_fp32_ref(x_shape, x, y);
}

}}}; // namespace ppl::kernel::x86
