// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for arithmeticitional information
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
#include "ppl/kernel/arm_server/arithmetic/neon/arithmetic_common.h"
#include "ppl/kernel/arm_server/arithmetic/neon/arithmetic_eltwise_common.h"
#include "ppl/kernel/arm_server/arithmetic/neon/arithmetic_broadcast_ndarray_common.h"
#include "ppl/kernel/arm_server/arithmetic/neon/arithmetic_broadcast_nbcx_common.h"

#include "ppl/kernel/arm_server/arithmetic/neon/arithmetic_kernel.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT, arithmetic_op_type_t op_type, bool fuse_relu>
ppl::common::RetCode arithmetic(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const eT *src0,
    const eT *src1,
    eT *dst)
{
    const bool is_eltwise = src0_shape->CalcElementsExcludingPadding() == dst_shape->CalcElementsExcludingPadding() &&
                            src1_shape->CalcElementsExcludingPadding() == dst_shape->CalcElementsExcludingPadding();
    if (is_eltwise) {
        return arithmetic_eltwise_common<eT, op_type, fuse_relu>(dst_shape, src0, src1, dst);
    }

    const auto data_format = src0_shape->GetDataFormat();
    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        return arithmetic_broadcast_ndarray_common<eT, op_type, fuse_relu>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
    }

    // NBCX
    if (std::is_same<eT, float>::value) {
        if (data_format == ppl::common::DATAFORMAT_N4CX) { // fp32 n4cx
            return arithmetic_broadcast_nbcx_common<float, 4, op_type, fuse_relu>(src0_shape, src1_shape, dst_shape, (const float *)src0, (const float *)src1, (float *)dst);
        }
    }
#ifdef PPLNN_USE_ARMV8_2_FP16
    if (std::is_same<eT, __fp16>::value) {
        if (data_format == ppl::common::DATAFORMAT_N8CX) { // fp16 n8cx
            return arithmetic_broadcast_nbcx_common<__fp16, 8, op_type, fuse_relu>(src0_shape, src1_shape, dst_shape, (const __fp16 *)src0, (const __fp16 *)src1, (__fp16 *)dst);
        }
    }
#endif

    return ppl::common::RC_UNSUPPORTED;
}

template <arithmetic_op_type_t op_type>
ppl::common::RetCode arithmetic_wrapper(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src0,
    const void *src1,
    const bool fuse_relu,
    void *dst)
{
    const auto data_type = src0_shape->GetDataType();
    if (fuse_relu) {
        switch (data_type) {
            case ppl::common::DATATYPE_FLOAT32: return arithmetic<float, op_type, true>(src0_shape, src1_shape, dst_shape, (const float *)src0, (const float *)src1, (float *)dst);
            case ppl::common::DATATYPE_INT64: return arithmetic<int64_t, op_type, true>(src0_shape, src1_shape, dst_shape, (const int64_t *)src0, (const int64_t *)src1, (int64_t *)dst);
#ifdef PPLNN_USE_ARMV8_2_FP16
            case ppl::common::DATATYPE_FLOAT16: return arithmetic<__fp16, op_type, true>(src0_shape, src1_shape, dst_shape, (const __fp16 *)src0, (const __fp16 *)src1, (__fp16 *)dst);
#endif
            default: break;
        }
    } else {
        switch (data_type) {
            case ppl::common::DATATYPE_FLOAT32: return arithmetic<float, op_type, false>(src0_shape, src1_shape, dst_shape, (const float *)src0, (const float *)src1, (float *)dst);
            case ppl::common::DATATYPE_INT64: return arithmetic<int64_t, op_type, false>(src0_shape, src1_shape, dst_shape, (const int64_t *)src0, (const int64_t *)src1, (int64_t *)dst);
#ifdef PPLNN_USE_ARMV8_2_FP16
            case ppl::common::DATATYPE_FLOAT16: return arithmetic<__fp16, op_type, false>(src0_shape, src1_shape, dst_shape, (const __fp16 *)src0, (const __fp16 *)src1, (__fp16 *)dst);
#endif
            default: break;
        }
    }
    return ppl::common::RC_UNSUPPORTED;
}

ppl::common::RetCode add(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src0,
    const void *src1,
    const bool fuse_relu,
    void *dst)
{
    return arithmetic_wrapper<ARITHMETIC_ADD>(src0_shape, src1_shape, dst_shape, src0, src1, fuse_relu, dst);
}

ppl::common::RetCode sub(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src0,
    const void *src1,
    const bool fuse_relu,
    void *dst)
{
    return arithmetic_wrapper<ARITHMETIC_SUB>(src0_shape, src1_shape, dst_shape, src0, src1, fuse_relu, dst);
}

ppl::common::RetCode mul(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src0,
    const void *src1,
    const bool fuse_relu,
    void *dst)
{
    return arithmetic_wrapper<ARITHMETIC_MUL>(src0_shape, src1_shape, dst_shape, src0, src1, fuse_relu, dst);
}

ppl::common::RetCode div(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src0,
    const void *src1,
    const bool fuse_relu,
    void *dst)
{
    return arithmetic_wrapper<ARITHMETIC_DIV>(src0_shape, src1_shape, dst_shape, src0, src1, fuse_relu, dst);
}

}}}}; // namespace ppl::kernel::arm_server::neon
