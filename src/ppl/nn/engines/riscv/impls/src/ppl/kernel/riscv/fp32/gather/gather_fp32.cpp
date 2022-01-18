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

#include "ppl/kernel/riscv/common/gather/gather_common.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode gather_ndarray_fp32(
    const float* src,
    float* dst,

    const int64_t* indices,
    const int64_t outer_dim,
    const int64_t gather_dim,
    const int64_t inner_dim,
    const int64_t num_indices,
    const int64_t indices_dim)
{
    return gather_ndarray_common<float>(src, dst, indices, outer_dim, gather_dim, inner_dim, num_indices, indices_dim);
}

}}} // namespace ppl::kernel::riscv
