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

#include "ppl/nn/utils/generic_cpu_device.h"
#include "ppl/nn/common/logger.h"
#include <cstring> // memcpy
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace utils {

RetCode GenericCpuDevice::Realloc(uint64_t bytes, BufferDesc* buffer) {
    if (buffer->addr) {
        allocator_.Free(buffer->addr);
    }

    if (bytes == 0) {
        buffer->addr = nullptr;
        return RC_SUCCESS;
    }

    buffer->addr = allocator_.Alloc(bytes);
    if (!buffer->addr) {
        return RC_OUT_OF_MEMORY;
    }

    buffer->desc = bytes;
    return RC_SUCCESS;
}

RetCode GenericCpuDevice::Realloc(const TensorShape& shape, BufferDesc* buffer) {
    return Realloc(shape.CalcBytesIncludingPadding(), buffer);
}

void GenericCpuDevice::Free(BufferDesc* buffer) {
    if (buffer->addr) {
        allocator_.Free(buffer->addr);
        buffer->addr = nullptr;
    }
}

RetCode GenericCpuDevice::CopyFromHost(BufferDesc* dst, const void* src, uint64_t bytes) const {
    memcpy(dst->addr, src, bytes);
    return RC_SUCCESS;
}

RetCode GenericCpuDevice::CopyFromHostAsync(BufferDesc* dst, const void* src, uint64_t bytes) const {
    return CopyFromHost(dst, src, bytes);
}

RetCode GenericCpuDevice::CopyFromHost(BufferDesc* dst, const void* src, const TensorShape& shape) const {
    return CopyFromHost(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode GenericCpuDevice::CopyFromHostAsync(BufferDesc* dst, const void* src, const TensorShape& shape) const {
    return CopyFromHostAsync(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode GenericCpuDevice::CopyToHost(void* dst, const BufferDesc& src, uint64_t bytes) const {
    memcpy(dst, src.addr, bytes);
    return RC_SUCCESS;
}

RetCode GenericCpuDevice::CopyToHostAsync(void* dst, const BufferDesc& src, uint64_t bytes) const {
    return CopyToHost(dst, src.addr, bytes);
}

RetCode GenericCpuDevice::CopyToHost(void* dst, const BufferDesc& src, const TensorShape& shape) const {
    return CopyToHost(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode GenericCpuDevice::CopyToHostAsync(void* dst, const BufferDesc& src, const TensorShape& shape) const {
    return CopyToHostAsync(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode GenericCpuDevice::Copy(BufferDesc* dst, const BufferDesc& src, uint64_t bytes) const {
    memcpy(dst->addr, src.addr, bytes);
    return RC_SUCCESS;
}

RetCode GenericCpuDevice::Copy(BufferDesc* dst, const BufferDesc& src, const TensorShape& shape) const {
    return Copy(dst, src, shape.CalcBytesIncludingPadding());
}

template <typename SrcType, typename DstType>
static void TypedConvert(DstType* dst, const SrcType* src, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        dst[i] = src[i];
    }
}

RetCode GenericCpuDevice::Convert(BufferDesc* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                  const TensorShape& src_desc, const void*, const void*) {
    if (dst_desc.GetDataFormat() != DATAFORMAT_NDARRAY) {
        LOG(ERROR) << "unsupported target format [" << GetDataFormatStr(dst_desc.GetDataFormat()) << "].";
        return RC_UNSUPPORTED;
    }

    if (src_desc.GetDataFormat() != DATAFORMAT_NDARRAY) {
        LOG(ERROR) << "unsupported target format [" << GetDataFormatStr(dst_desc.GetDataFormat()) << "].";
        return RC_UNSUPPORTED;
    }

    if (dst_desc.GetDataType() == src_desc.GetDataType()) {
        memcpy(dst->addr, src.addr, src_desc.CalcBytesIncludingPadding());
        return RC_SUCCESS;
    }

    if (dst_desc.GetDataType() == DATATYPE_FLOAT32) {
        if (src_desc.GetDataType() == DATATYPE_INT64) {
            TypedConvert(static_cast<float*>(dst->addr), static_cast<const int64_t*>(src.addr),
                         src_desc.CalcElementsIncludingPadding());
            return RC_SUCCESS;
        }
    } else if (dst_desc.GetDataType() == DATATYPE_INT64) {
        if (src_desc.GetDataType() == DATATYPE_FLOAT32) {
            TypedConvert(static_cast<int64_t*>(dst->addr), static_cast<const float*>(src.addr),
                         src_desc.CalcElementsIncludingPadding());
            return RC_SUCCESS;
        }
    }

    LOG(ERROR) << "unsupported source data type [" << GetDataTypeStr(src_desc.GetDataType())
               << "] to target data type [" << GetDataTypeStr(dst_desc.GetDataType()) << "].";
    return RC_UNSUPPORTED;
}

RetCode GenericCpuDevice::ConvertToHost(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                        const TensorShape& src_desc, const void*) {
    BufferDesc dst_wrapper(dst);
    return Convert(&dst_wrapper, dst_desc, src, src_desc);
}

RetCode GenericCpuDevice::ConvertToHostAsync(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                             const TensorShape& src_desc, const void* src_info) {
    return ConvertToHost(dst, dst_desc, src, src_desc, src_info);
}

RetCode GenericCpuDevice::ConvertFromHost(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                          const TensorShape& src_desc, const void*) {
    return Convert(dst, dst_desc, BufferDesc(const_cast<void*>(src)), src_desc);
}

RetCode GenericCpuDevice::ConvertFromHostAsync(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                               const TensorShape& src_desc, const void* dst_info) {
    return ConvertFromHost(dst, dst_desc, src, src_desc, dst_info);
}

}}} // namespace ppl::nn::utils
