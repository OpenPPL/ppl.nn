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

#include "ppl/kernel/x86/common/internal_include.h"
#include <string.h>

namespace ppl { namespace kernel { namespace x86 {

template <typename srcT, typename dstT>
ppl::common::RetCode cast_kernel(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const srcT *src,
    dstT *dst)
{
    const bool out_bool   = dst_shape->GetDataType() == ppl::common::DATATYPE_BOOL;
    const uint64_t length = src_shape->GetElementsIncludingPadding();

    if (out_bool) {
        PRAGMA_OMP_PARALLEL_FOR()
        for (uint64_t i = 0; i < length; i++) {
            dst[i] = src[i] != 0 ? 1 : 0;
        }
    } else {
        PRAGMA_OMP_PARALLEL_FOR()
        for (uint64_t i = 0; i < length; i++) {
            dst[i] = src[i];
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode cast(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const void *src,
    void *dst)
{
#define MAKE_CAST_TYPE(idt, odt) (((uint32_t)idt << 16) | (uint32_t)odt)

    auto idt = src_shape->GetDataType();
    auto odt = dst_shape->GetDataType();

    if (idt == odt) {
        memcpy(dst, src, dst_shape->GetBytesIncludingPadding());
        return ppl::common::RC_SUCCESS;
    }

    switch (MAKE_CAST_TYPE(idt, odt)) {
        case MAKE_CAST_TYPE(ppl::common::DATATYPE_FLOAT32, ppl::common::DATATYPE_INT64):
            return cast_kernel<float, int64_t>(src_shape, dst_shape, (float*)src, (int64_t*)dst);
        case MAKE_CAST_TYPE(ppl::common::DATATYPE_FLOAT32, ppl::common::DATATYPE_BOOL):
            return cast_kernel<float, uint8_t>(src_shape, dst_shape, (float*)src, (uint8_t*)dst);
        case MAKE_CAST_TYPE(ppl::common::DATATYPE_INT64, ppl::common::DATATYPE_FLOAT32):
            return cast_kernel<int64_t, float>(src_shape, dst_shape, (int64_t*)src, (float*)dst);
        case MAKE_CAST_TYPE(ppl::common::DATATYPE_INT64, ppl::common::DATATYPE_BOOL):
            return cast_kernel<int64_t, uint8_t>(src_shape, dst_shape, (int64_t*)src, (uint8_t*)dst);
        case MAKE_CAST_TYPE(ppl::common::DATATYPE_BOOL, ppl::common::DATATYPE_FLOAT32):
            return cast_kernel<uint8_t, float>(src_shape, dst_shape, (uint8_t*)src, (float*)dst);
        case MAKE_CAST_TYPE(ppl::common::DATATYPE_BOOL, ppl::common::DATATYPE_INT64):
            return cast_kernel<uint8_t, int64_t>(src_shape, dst_shape, (uint8_t*)src, (int64_t*)dst);
        case MAKE_CAST_TYPE(ppl::common::DATATYPE_FLOAT32, ppl::common::DATATYPE_FLOAT64):
            return cast_kernel<float, double>(src_shape, dst_shape, (float*)src, (double*)dst);
        case MAKE_CAST_TYPE(ppl::common::DATATYPE_FLOAT64, ppl::common::DATATYPE_FLOAT32):
            return cast_kernel<double, float>(src_shape, dst_shape, (double*)src, (float*)dst);
        default:
            return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}}; // namespace ppl::kernel::x86
