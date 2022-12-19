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

#include <string.h>
#include <stdint.h>

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/pad/neon/pad_common.h"
#include "ppl/kernel/arm_server/pad/neon/pad_ndarray_common.h"
#include "ppl/kernel/arm_server/pad/neon/pad_nbcx_common.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT, pad_mode_type_t _mode>
static ppl::common::RetCode pad_wrapper(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const eT *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    const eT *constant_value,
    eT *dst)
{
    const eT constant_val = constant_value == nullptr ? (eT)0 : *constant_value;

    const auto data_format = src_shape->GetDataFormat();
    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        return pad_ndarray_common<eT, _mode>(src_shape, dst_shape, src, start_pads, end_pads, constant_val, dst);
    }

    // NBCX
    if (sizeof(eT) == 4) {
        if (data_format == ppl::common::DATAFORMAT_N4CX) { // fp32 n4cx
            return pad_nbcx_common<uint32_t, 4, _mode>(src_shape, dst_shape, (const uint32_t *)src, start_pads, end_pads, (const eT)constant_val, (uint32_t *)dst);
        }
    }
    if (sizeof(eT) == 2) {
        if (data_format == ppl::common::DATAFORMAT_N8CX) { // fp16 n8cx
            return pad_nbcx_common<uint16_t, 8, _mode>(src_shape, dst_shape, (const uint16_t *)src, start_pads, end_pads, (const eT)constant_val, (uint16_t *)dst);
        }
    }

    return ppl::common::RC_UNSUPPORTED;
}

template <pad_mode_type_t _mode>
static ppl::common::RetCode pad(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    const void *constant_value,
    void *dst)
{
    const auto data_type = src_shape->GetDataType();
    switch (ppl::common::GetSizeOfDataType(data_type)) {
        case 1: return pad_wrapper<uint8_t, _mode>(src_shape, dst_shape, (const uint8_t *)src, start_pads, end_pads, (const uint8_t *)constant_value, (uint8_t *)dst);
        case 2: return pad_wrapper<uint16_t, _mode>(src_shape, dst_shape, (const uint16_t *)src, start_pads, end_pads, (const uint16_t *)constant_value, (uint16_t *)dst);
        case 4: return pad_wrapper<uint32_t, _mode>(src_shape, dst_shape, (const uint32_t *)src, start_pads, end_pads, (const uint32_t *)constant_value, (uint32_t *)dst);
        case 8: return pad_wrapper<uint64_t, _mode>(src_shape, dst_shape, (const uint64_t *)src, start_pads, end_pads, (const uint64_t *)constant_value, (uint64_t *)dst);
        default: break;
    }

    return ppl::common::RC_UNSUPPORTED;
}

ppl::common::RetCode pad_constant(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    const void *constant_value,
    void *dst)
{
    return pad<PAD_MODE_CONSTANT>(src_shape, dst_shape, src, start_pads, end_pads, constant_value, dst);
}

ppl::common::RetCode pad_reflect(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    const void *constant_value,
    void *dst)
{
    return pad<PAD_MODE_REFLECT>(src_shape, dst_shape, src, start_pads, end_pads, constant_value, dst);
}

ppl::common::RetCode pad_edge(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    const void *constant_value,
    void *dst)
{
    return pad<PAD_MODE_EDGE>(src_shape, dst_shape, src, start_pads, end_pads, constant_value, dst);
}

}}}}; // namespace ppl::kernel::arm_server::neon
