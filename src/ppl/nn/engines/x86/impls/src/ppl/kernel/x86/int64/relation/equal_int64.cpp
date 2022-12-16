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

#include "relation_int64_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode equal_eltwise_int64(
    const ppl::common::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    uint8_t *dst)
{
    return relation_eltwise_binary_op_int64<RELATION_EQUAL>(dst_shape, src0, src1, dst);
}

ppl::common::RetCode equal_ndarray_int64(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    uint8_t *dst)
{
    return relation_ndarray_binary_op_int64<RELATION_EQUAL>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
}

}}}; // namespace ppl::kernel::x86