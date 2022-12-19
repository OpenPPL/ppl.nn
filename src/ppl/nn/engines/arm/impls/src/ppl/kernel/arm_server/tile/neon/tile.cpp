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
#include "ppl/kernel/arm_server/tile/neon/tile_common.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

ppl::common::RetCode tile(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src,
    const void *repeats,
    void *dst)
{
    const auto data_type = src_shape->GetDataType();

    switch (ppl::common::GetSizeOfDataType(data_type)) {
        case 1: return tile_ndarray_common<uint8_t>(src_shape, dst_shape, (const uint8_t *)src, (const int64_t *)repeats, (uint8_t *)dst);
        case 2: return tile_ndarray_common<uint16_t>(src_shape, dst_shape, (const uint16_t *)src, (const int64_t *)repeats, (uint16_t *)dst);
        case 4: return tile_ndarray_common<uint32_t>(src_shape, dst_shape, (const uint32_t *)src, (const int64_t *)repeats, (uint32_t *)dst);
        case 8: return tile_ndarray_common<uint64_t>(src_shape, dst_shape, (const uint64_t *)src, (const int64_t *)repeats, (uint64_t *)dst);
        default: break;
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}}}; // namespace ppl::kernel::arm_server::neon
