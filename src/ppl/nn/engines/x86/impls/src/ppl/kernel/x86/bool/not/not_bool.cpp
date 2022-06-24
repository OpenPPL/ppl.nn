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

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode not_bool(
    const ppl::nn::TensorShape *x_shape,
    const uint8_t *x,
    uint8_t *y)
{
    const int64_t n_elem = x_shape->CalcElementsIncludingPadding();

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < n_elem; ++i) {
        y[i] = x[i] ^ 0x01;
    }
    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86