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
#include "ppl/kernel/riscv/common/internal_include.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode exp_fp32(
    const ppl::common::TensorShape* src_shape,
    const float* src,
    float* dst)
{
#define _OP_SS(Y, X) \
    do {             \
        Y = expf(X); \
    } while (0)

    const int64_t n_elem      = src_shape->CalcElementsIncludingPadding();
    const int64_t unroll_n    = 16;
    const int64_t unroll_body = round(n_elem, unroll_n);

    for (int64_t i = 0; i < unroll_body; i += unroll_n) {
        _OP_SS(dst[i + 0], src[i + 0]);
        _OP_SS(dst[i + 1], src[i + 1]);
        _OP_SS(dst[i + 2], src[i + 2]);
        _OP_SS(dst[i + 3], src[i + 3]);
        _OP_SS(dst[i + 4], src[i + 4]);
        _OP_SS(dst[i + 5], src[i + 5]);
        _OP_SS(dst[i + 6], src[i + 6]);
        _OP_SS(dst[i + 7], src[i + 7]);
        _OP_SS(dst[i + 8], src[i + 8]);
        _OP_SS(dst[i + 9], src[i + 9]);
        _OP_SS(dst[i + 10], src[i + 10]);
        _OP_SS(dst[i + 11], src[i + 11]);
        _OP_SS(dst[i + 12], src[i + 12]);
        _OP_SS(dst[i + 13], src[i + 13]);
        _OP_SS(dst[i + 14], src[i + 14]);
        _OP_SS(dst[i + 15], src[i + 15]);
    }
    for (int64_t i = unroll_body; i < n_elem; i++) {
        _OP_SS(dst[i], src[i]);
    }
#undef _OP_SS
    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv