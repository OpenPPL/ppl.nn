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

#include "arithmetic_eltwise_int64.h"
#include "arithmetic_broadcast_ndarray_int64.h"
#include "arithmetic_broadcast_n16cx_int64.h"

namespace ppl { namespace kernel { namespace x86 {

template <arithmetic_op_type_t _op>
static ppl::common::RetCode arithmetic_int64(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    int64_t *dst)
{
    bool is_eltwise =
        src0_shape->GetElementsExcludingPadding() == dst_shape->GetElementsExcludingPadding() &&
        src1_shape->GetElementsExcludingPadding() == dst_shape->GetElementsExcludingPadding();
    if (is_eltwise) {
        return arithmetic_eltwise_int64<_op>(dst_shape, src0, src1, dst);
    } else if (dst_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        return arithmetic_broadcast_ndarray_int64<_op>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
    } else if (dst_shape->GetDataFormat() == ppl::common::DATAFORMAT_N16CX) {
        return arithmetic_broadcast_n16cx_int64<_op>(src0_shape, src1_shape, dst_shape, src0, src1, 1, dst);
    }

    return ppl::common::RC_UNSUPPORTED;
}

ppl::common::RetCode add_int64(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    int64_t *dst)
{
    return arithmetic_int64<ARITHMETIC_ADD>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
}

ppl::common::RetCode sub_int64(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    int64_t *dst)
{
    return arithmetic_int64<ARITHMETIC_SUB>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
}

ppl::common::RetCode mul_int64(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    int64_t *dst)
{
    return arithmetic_int64<ARITHMETIC_MUL>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
}

ppl::common::RetCode div_int64(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    int64_t *dst)
{
    return arithmetic_int64<ARITHMETIC_DIV>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
}

}}}; // namespace ppl::kernel::x86