// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for relationitional information
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

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/relation/neon/relation_common.h"
#include "ppl/kernel/arm_server/relation/neon/relation_eltwise_common.h"
#include "ppl/kernel/arm_server/relation/neon/relation_broadcast_ndarray_common.h"
#include "ppl/kernel/arm_server/relation/neon/relation_broadcast_common.h"


#include "ppl/kernel/arm_server/relation/neon/relation_kernel.h"
namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT, relation_op_type_t op_type>
ppl::common::RetCode relation(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const eT *src0,
    const eT *src1,
    uint8_t *dst)
{
    const bool is_eltwise = src0_shape->CalcElementsExcludingPadding() == dst_shape->CalcElementsExcludingPadding() &&
                            src1_shape->CalcElementsExcludingPadding() == dst_shape->CalcElementsExcludingPadding();
    if (is_eltwise) {
        return relation_eltwise_common<eT, op_type>(dst_shape, src0, src1, dst);
    }

    const auto data_format = src0_shape->GetDataFormat();
    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        return relation_broadcast_ndarray_common<eT, op_type>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
    }

    // NBCX
    if (std::is_same<eT, float>::value) {
        if (data_format == ppl::common::DATAFORMAT_N4CX) { // fp32 n4cx
            return relation_broadcast_common<float, 4, op_type>(src0_shape, src1_shape, dst_shape, (const float *)src0, (const float *)src1, (uint8_t* )dst);
        }
    }
#ifdef PPLNN_USE_ARMV8_2_FP16
    if (std::is_same<eT, __fp16>::value) {
        if (data_format == ppl::common::DATAFORMAT_N8CX) { // fp16 n8cx
            return relation_broadcast_common<__fp16, 8, op_type>(src0_shape, src1_shape, dst_shape, (const __fp16 *)src0, (const __fp16 *)src1, (uint8_t* )dst);
        }
    }
#endif

    return ppl::common::RC_UNSUPPORTED;
}

template <relation_op_type_t op_type>
ppl::common::RetCode relation_wrapper(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src0,
    const void *src1,
    uint8_t *dst)
{
    const auto data_type = src0_shape->GetDataType();
    switch (data_type) {
        case ppl::common::DATATYPE_FLOAT32: return relation<float, op_type>(src0_shape, src1_shape, dst_shape, (const float *)src0, (const float *)src1, dst);
        case ppl::common::DATATYPE_INT64: return relation<int64_t, op_type>(src0_shape, src1_shape, dst_shape, (const int64_t *)src0, (const int64_t *)src1, dst);
#ifdef PPLNN_USE_ARMV8_2_FP16
        case ppl::common::DATATYPE_FLOAT16: return relation<__fp16, op_type>(src0_shape, src1_shape, dst_shape, (const __fp16 *)src0, (const __fp16 *)src1, dst);
#endif
        default: break;
    }
    return ppl::common::RC_UNSUPPORTED;
}

ppl::common::RetCode equal(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src0,
    const void *src1,
    uint8_t *dst)
{
    return relation_wrapper<RELATION_EQUAL>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
}

ppl::common::RetCode less(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src0,
    const void *src1,
    uint8_t *dst)
{
    return relation_wrapper<RELATION_LESS>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
}

}}}}; // namespace ppl::kernel::arm_server::neon
