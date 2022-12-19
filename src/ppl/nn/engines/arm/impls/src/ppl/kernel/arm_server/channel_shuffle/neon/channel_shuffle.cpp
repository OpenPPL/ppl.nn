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
#include "ppl/kernel/arm_server/channel_shuffle/neon/channel_shuffle_ndarray_common.h"
#include "ppl/kernel/arm_server/channel_shuffle/neon/channel_shuffle_nbcx_common.h"
#include "ppl/kernel/arm_server/channel_shuffle/neon/channel_shuffle_specializations.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT>
static ppl::common::RetCode channel_shuffle_concat_split_wrapper(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst0_shape,
    const ppl::common::TensorShape *dst1_shape,
    const void *src0,
    const void *src1,
    const int32_t group,
    void *dst0,
    void *dst1)
{
    const auto data_format = src0_shape->GetDataFormat();
    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        return channel_shuffle_ndarray_concat_split_common<eT>(src0_shape, src1_shape, dst0_shape, dst1_shape, (const eT *)src0, (const eT *)src1, group, (eT *)dst0, (eT *)dst1);
    }

    // NBCX
    if (sizeof(eT) == 4) {
        if (data_format == ppl::common::DATAFORMAT_N4CX) { // fp32 n4cx
            auto status = channel_shuffle_nbcx_concat_split_same2group_common<uint32_t, 4>(src0_shape, src1_shape, dst0_shape, dst1_shape, (const uint32_t *)src0, (const uint32_t *)src1, group, (uint32_t *)dst0, (uint32_t *)dst1);
            if (status == ppl::common::RC_SUCCESS) {
                return status;
            }
            return channel_shuffle_nbcx_concat_split_common<uint32_t, 4>(src0_shape, src1_shape, dst0_shape, dst1_shape, (const uint32_t *)src0, (const uint32_t *)src1, group, (uint32_t *)dst0, (uint32_t *)dst1);
        }
    }
    if (sizeof(eT) == 2) {
        if (data_format == ppl::common::DATAFORMAT_N8CX) { // fp16 n8cx
            auto status = channel_shuffle_nbcx_concat_split_same2group_common<uint16_t, 8>(src0_shape, src1_shape, dst0_shape, dst1_shape, (const uint16_t *)src0, (const uint16_t *)src1, group, (uint16_t *)dst0, (uint16_t *)dst1);
            if (status == ppl::common::RC_SUCCESS) {
                return status;
            }
            return channel_shuffle_nbcx_concat_split_common<uint16_t, 8>(src0_shape, src1_shape, dst0_shape, dst1_shape, (const uint16_t *)src0, (const uint16_t *)src1, group, (uint16_t *)dst0, (uint16_t *)dst1);
        }
    }

    return ppl::common::RC_UNSUPPORTED;
}

ppl::common::RetCode channel_shuffle_concat_split(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst0_shape,
    const ppl::common::TensorShape *dst1_shape,
    const void *src0,
    const void *src1,
    const int32_t group,
    void *dst0,
    void *dst1)
{
    const auto data_type = src0_shape->GetDataType();
    switch (ppl::common::GetSizeOfDataType(data_type)) {
        case 1: return channel_shuffle_concat_split_wrapper<uint8_t>(src0_shape, src1_shape, dst0_shape, dst1_shape, src0, src1, group, dst0, dst1);
        case 2: return channel_shuffle_concat_split_wrapper<uint16_t>(src0_shape, src1_shape, dst0_shape, dst1_shape, src0, src1, group, dst0, dst1);
        case 4: return channel_shuffle_concat_split_wrapper<uint32_t>(src0_shape, src1_shape, dst0_shape, dst1_shape, src0, src1, group, dst0, dst1);
        case 8: return channel_shuffle_concat_split_wrapper<uint64_t>(src0_shape, src1_shape, dst0_shape, dst1_shape, src0, src1, group, dst0, dst1);
        default: break;
    }
    return ppl::common::RC_UNSUPPORTED;
}

template <typename eT>
static ppl::common::RetCode channel_shuffle_concat_wrapper(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src0,
    const void *src1,
    const int32_t group,
    void *dst)
{
    const auto data_format = src0_shape->GetDataFormat();
    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        return channel_shuffle_ndarray_concat_common<eT>(src0_shape, src1_shape, dst_shape, (const eT *)src0, (const eT *)src1, group, (eT *)dst);
    }

    // NBCX
    if (sizeof(eT) == 4) {
        if (data_format == ppl::common::DATAFORMAT_N4CX) { // fp32 n4cx
            return channel_shuffle_nbcx_concat_common<uint32_t, 4>(src0_shape, src1_shape, dst_shape, (const uint32_t *)src0, (const uint32_t *)src1, group, (uint32_t *)dst);
        }
    }
    if (sizeof(eT) == 2) {
        if (data_format == ppl::common::DATAFORMAT_N8CX) { // fp16 n8cx
            return channel_shuffle_nbcx_concat_common<uint16_t, 8>(src0_shape, src1_shape, dst_shape, (const uint16_t *)src0, (const uint16_t *)src1, group, (uint16_t *)dst);
        }
    }

    return ppl::common::RC_UNSUPPORTED;
}

ppl::common::RetCode channel_shuffle_concat(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src0,
    const void *src1,
    const int32_t group,
    void *dst)
{
    const auto data_type = src0_shape->GetDataType();
    switch (ppl::common::GetSizeOfDataType(data_type)) {
        case 1: return channel_shuffle_concat_wrapper<uint8_t>(src0_shape, src1_shape, dst_shape, src0, src1, group, dst);
        case 2: return channel_shuffle_concat_wrapper<uint16_t>(src0_shape, src1_shape, dst_shape, src0, src1, group, dst);
        case 4: return channel_shuffle_concat_wrapper<uint32_t>(src0_shape, src1_shape, dst_shape, src0, src1, group, dst);
        case 8: return channel_shuffle_concat_wrapper<uint64_t>(src0_shape, src1_shape, dst_shape, src0, src1, group, dst);
        default: break;
    }
    return ppl::common::RC_UNSUPPORTED;
}

template <typename eT>
static ppl::common::RetCode channel_shuffle_wrapper(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src,
    const int32_t group,
    void *dst)
{
    const auto data_format = src_shape->GetDataFormat();
    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        return channel_shuffle_ndarray_common<eT>(src_shape, dst_shape, (const eT *)src, group, (eT *)dst);
    }

    // NBCX
    if (sizeof(eT) == 4) {
        if (data_format == ppl::common::DATAFORMAT_N4CX) { // fp32 n4cx
            return channel_shuffle_nbcx_common<uint32_t, 4>(src_shape, dst_shape, (const uint32_t *)src, group, (uint32_t *)dst);
        }
    }
    if (sizeof(eT) == 2) {
        if (data_format == ppl::common::DATAFORMAT_N8CX) { // fp16 n8cx
            return channel_shuffle_nbcx_common<uint16_t, 8>(src_shape, dst_shape, (const uint16_t *)src, group, (uint16_t *)dst);
        }
    }

    return ppl::common::RC_UNSUPPORTED;
}

ppl::common::RetCode channel_shuffle(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src,
    const int32_t group,
    void *dst)
{
    const auto data_type = src_shape->GetDataType();
    switch (ppl::common::GetSizeOfDataType(data_type)) {
        case 1: return channel_shuffle_wrapper<uint8_t>(src_shape, dst_shape, src, group, dst);
        case 2: return channel_shuffle_wrapper<uint16_t>(src_shape, dst_shape, src, group, dst);
        case 4: return channel_shuffle_wrapper<uint32_t>(src_shape, dst_shape, src, group, dst);
        case 8: return channel_shuffle_wrapper<uint64_t>(src_shape, dst_shape, src, group, dst);
        default: break;
    }
    return ppl::common::RC_UNSUPPORTED;
}

}}}} // namespace ppl::kernel::arm_server::neon
