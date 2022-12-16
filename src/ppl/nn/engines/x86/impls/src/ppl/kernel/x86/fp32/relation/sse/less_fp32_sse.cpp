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

#include "relation_fp32_sse_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode less_eltwise_fp32_sse(
    const ppl::common::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    uint8_t *dst)
{
    return relation_eltwise_binary_op_fp32_sse<RELATION_LESS>(dst_shape, src0, src1, dst);
}

ppl::common::RetCode less_ndarray_fp32_sse(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    uint8_t *dst)
{
    return relation_ndarray_binary_op_fp32_sse<RELATION_LESS>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
}

}}}; // namespace ppl::kernel::x86