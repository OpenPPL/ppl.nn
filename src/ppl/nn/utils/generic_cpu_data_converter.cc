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

#include "ppl/nn/utils/generic_cpu_data_converter.h"
#include <cstring> // memcpy
using namespace ppl::common;

namespace ppl { namespace nn { namespace utils {

template <typename SrcType, typename DstType>
static void TypedConvert(DstType* dst, const SrcType* src, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        dst[i] = src[i];
    }
}

RetCode GenericCpuDataConverter::Convert(BufferDesc* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                         const TensorShape& src_desc) const {
    if (dst_desc.GetDataFormat() != DATAFORMAT_NDARRAY || src_desc.GetDataFormat() != DATAFORMAT_NDARRAY) {
        return RC_UNSUPPORTED;
    }

    if (dst_desc.GetDataType() == src_desc.GetDataType()) {
        memcpy(dst->addr, src.addr, src_desc.GetBytesIncludingPadding());
        return RC_SUCCESS;
    }

    if (dst_desc.GetDataType() == DATATYPE_FLOAT32) {
        if (src_desc.GetDataType() == DATATYPE_INT64) {
            TypedConvert(static_cast<float*>(dst->addr), static_cast<const int64_t*>(src.addr),
                         src_desc.GetElementsIncludingPadding());
            return RC_SUCCESS;
        }
    } else if (dst_desc.GetDataType() == DATATYPE_INT64) {
        if (src_desc.GetDataType() == DATATYPE_FLOAT32) {
            TypedConvert(static_cast<int64_t*>(dst->addr), static_cast<const float*>(src.addr),
                         src_desc.GetElementsIncludingPadding());
            return RC_SUCCESS;
        }
    }

    return RC_UNSUPPORTED;
}

RetCode GenericCpuDataConverter::ConvertToHost(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                               const TensorShape& src_desc) const {
    BufferDesc dst_wrapper(dst);
    return Convert(&dst_wrapper, dst_desc, src, src_desc);
}

RetCode GenericCpuDataConverter::ConvertFromHost(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                                 const TensorShape& src_desc) const {
    return Convert(dst, dst_desc, BufferDesc(const_cast<void*>(src)), src_desc);
}

}}} // namespace ppl::nn::utils
