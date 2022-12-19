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

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/where/neon/where_common.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT>
ppl::common::RetCode where_common(
    const ppl::common::TensorShape *cond_shape,
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const uint8_t *cond,
    const eT *src0,
    const eT *src1,
    eT *dst)
{
    const bool is_eltwise = src0_shape->CalcElementsExcludingPadding() == dst_shape->CalcElementsExcludingPadding() &&
                            src1_shape->CalcElementsExcludingPadding() == dst_shape->CalcElementsExcludingPadding();

    if (is_eltwise) {
        return where_eltwise_common<eT>(dst_shape, cond, src0, src1, dst);
    }

    const auto data_format = src0_shape->GetDataFormat();
    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        return where_ndarray_common<eT>(cond_shape, src0_shape, src1_shape, dst_shape, cond, src0, src1, dst);
    }

    return ppl::common::RC_UNSUPPORTED;
}

ppl::common::RetCode where(
    const ppl::common::TensorShape *cond_shape,
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *cond,
    const void *src0,
    const void *src1,
    void *dst)
{
    const auto data_type = src0_shape->GetDataType();

    switch (ppl::common::GetSizeOfDataType(data_type)) {
        case 1: return where_common<uint8_t>(cond_shape, src0_shape, src1_shape, dst_shape, (const uint8_t *)cond, (const uint8_t *)src0, (const uint8_t *)src1, (uint8_t *)dst);
        case 2: return where_common<uint16_t>(cond_shape, src0_shape, src1_shape, dst_shape, (const uint8_t *)cond, (const uint16_t *)src0, (const uint16_t *)src1, (uint16_t *)dst);
        case 4: return where_common<uint32_t>(cond_shape, src0_shape, src1_shape, dst_shape, (const uint8_t *)cond, (const uint32_t *)src0, (const uint32_t *)src1, (uint32_t *)dst);
        case 8: return where_common<uint64_t>(cond_shape, src0_shape, src1_shape, dst_shape, (const uint8_t *)cond, (const uint64_t *)src0, (const uint64_t *)src1, (uint64_t *)dst);
        default: break;
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}}}; // namespace ppl::kernel::arm_server::neon
