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

#include "arithmetic_fp32_sse_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode div_ndarray_max6d_fp32_sse(
    const ppl::nn::TensorShape *lhs_shape,
    const ppl::nn::TensorShape *rhs_shape,
    const float *lhs,
    const float *rhs,
    float *dst)
{
    return arithmetic_binary_op_ndarray_fp32_sse<ARITHMETIC_DIV>(lhs_shape, rhs_shape, lhs, rhs, dst);
}

}}}; // namespace ppl::kernel::x86
