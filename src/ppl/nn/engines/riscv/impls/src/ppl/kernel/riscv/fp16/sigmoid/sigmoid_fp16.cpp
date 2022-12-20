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

ppl::common::RetCode sigmoid_fp16(const ppl::common::TensorShape* x_shape, const __fp16* x, __fp16* y)
{
    const int64_t n_elem   = x_shape->CalcElementsIncludingPadding();
    const int64_t n_unroll = 8;

    int64_t i = 0;
    for (; i <= n_elem - n_unroll; i += n_unroll) {
        y[i + 0] = 1.0f / (expf(-(float)x[i + 0]) + 1.0f);
        y[i + 1] = 1.0f / (expf(-(float)x[i + 1]) + 1.0f);
        y[i + 2] = 1.0f / (expf(-(float)x[i + 2]) + 1.0f);
        y[i + 3] = 1.0f / (expf(-(float)x[i + 3]) + 1.0f);
        y[i + 4] = 1.0f / (expf(-(float)x[i + 4]) + 1.0f);
        y[i + 5] = 1.0f / (expf(-(float)x[i + 5]) + 1.0f);
        y[i + 6] = 1.0f / (expf(-(float)x[i + 6]) + 1.0f);
        y[i + 7] = 1.0f / (expf(-(float)x[i + 7]) + 1.0f);
    }
    for (; i < n_elem; ++i) {
        y[i] = 1.0f / (expf(-(float)x[i]) + 1.0f);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv
