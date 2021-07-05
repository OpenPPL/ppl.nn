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
    return Realloc(shape.GetBytesIncludingPadding(), buffer);
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

RetCode GenericCpuDevice::CopyFromHost(BufferDesc* dst, const void* src, const TensorShape& shape) const {
    return CopyFromHost(dst, src, shape.GetBytesIncludingPadding());
}

RetCode GenericCpuDevice::CopyToHost(void* dst, const BufferDesc& src, uint64_t bytes) const {
    memcpy(dst, src.addr, bytes);
    return RC_SUCCESS;
}

RetCode GenericCpuDevice::CopyToHost(void* dst, const BufferDesc& src, const TensorShape& shape) const {
    return CopyToHost(dst, src, shape.GetBytesIncludingPadding());
}

RetCode GenericCpuDevice::Copy(BufferDesc* dst, const BufferDesc& src, uint64_t bytes) const {
    memcpy(dst->addr, src.addr, bytes);
    return RC_SUCCESS;
}

RetCode GenericCpuDevice::Copy(BufferDesc* dst, const BufferDesc& src, const TensorShape& shape) const {
    return Copy(dst, src, shape.GetBytesIncludingPadding());
}

}}} // namespace ppl::nn::utils
