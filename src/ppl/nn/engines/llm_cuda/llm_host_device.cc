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

#include "llm_host_device.h"

#include "ppl/nn/engines/llm_cuda/options.h"
#include "ppl/nn/utils/stack_buffer_manager.h"
#include "ppl/nn/utils/buffered_cpu_allocator.h"
#include "ppl/common/generic_cpu_allocator.h"
#include "ppl/nn/common/logger.h"

#include <stdarg.h>
#include <string.h>

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace llm { namespace cuda {

RetCode LlmHostDevice::Init(uint32_t mm_policy) {
    mm_policy_ = mm_policy;  
    if (mm_policy_ == MM_PLAIN) {
        allocator_ = std::shared_ptr<Allocator>(new GenericCpuAllocator(alignment_));
    } else if (mm_policy_ == MM_BESTFIT) {
        allocator_ = std::shared_ptr<Allocator>(new GenericCpuAllocator(alignment_));
        buffer_manager_.reset(new utils::StackBufferManager(allocator_.get()));
    } else if (mm_policy_ == MM_COMPACT) {
        auto allocator = new utils::BufferedCpuAllocator();
        auto rc = allocator->Init();
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "init BufferedCpuAllocator failed: " << GetRetCodeStr(rc);
            delete allocator;
            return rc;
        }

        vmr_.reset(allocator);
        buffer_manager_.reset(new utils::CompactBufferManager(allocator, alignment_));
    } else {
        LOG(ERROR) << "unsupported mm policy [" << mm_policy_ << "]";
        return RC_INVALID_VALUE;
    }

    return RC_SUCCESS;
}

RetCode LlmHostDevice::Realloc(uint64_t bytes, BufferDesc* buffer) {
    if (mm_policy_ == MM_PLAIN) {
        if (buffer->addr && buffer->desc == bytes)
            return RC_SUCCESS;
        
        if (buffer->addr)
            allocator_->Free(buffer->addr);

        if (bytes == 0) {
            buffer->addr = nullptr;
            return ppl::common::RC_SUCCESS;
        }

        buffer->addr = allocator_->Alloc(bytes);
        if (!buffer->addr) {
            return ppl::common::RC_OUT_OF_MEMORY;
        }

        buffer->desc = bytes;
        return ppl::common::RC_SUCCESS;
    } else {
        return buffer_manager_->Realloc(bytes, buffer);
    }
}

void LlmHostDevice::Free(BufferDesc* buffer) {
    if (mm_policy_ == MM_PLAIN) {
        if (buffer->addr) {
            allocator_->Free(buffer->addr);
            buffer->addr = nullptr;
        }
    } else {
        buffer_manager_->Free(buffer);
    }
}

LlmHostDevice::~LlmHostDevice() {
    LOG(DEBUG) << "buffer manager[" << buffer_manager_->GetName() << "] allocates ["
               << buffer_manager_->GetBufferedBytes() << "] bytes.";
    buffer_manager_.reset();
}

RetCode LlmHostDevice::CopyFromHostAsync(BufferDesc* dst, const void* src, uint64_t bytes) const {
    memcpy(dst->addr, src, bytes);
    return RC_SUCCESS;
}

RetCode LlmHostDevice::CopyFromHost(BufferDesc* dst, const void* src, uint64_t bytes) const {
    auto rc = CopyFromHostAsync(dst, src, bytes);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "CopyFromHostAsync failed";
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}

RetCode LlmHostDevice::CopyFromHost(BufferDesc* dst, const void* src, const TensorShape& shape) const {
    return CopyFromHost(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode LlmHostDevice::CopyFromHostAsync(BufferDesc* dst, const void* src, const TensorShape& shape) const {
    return CopyFromHostAsync(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode LlmHostDevice::CopyToHostAsync(void* dst, const BufferDesc& src, uint64_t bytes) const {
    memcpy(dst, src.addr, bytes);
    return RC_SUCCESS;
}

RetCode LlmHostDevice::CopyToHost(void* dst, const BufferDesc& src, uint64_t bytes) const {
    auto rc = CopyToHostAsync(dst, src, bytes);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "CopyToHostAsync failed";
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}

RetCode LlmHostDevice::CopyToHost(void* dst, const BufferDesc& src, const TensorShape& shape) const {
    return CopyToHost(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode LlmHostDevice::CopyToHostAsync(void* dst, const BufferDesc& src, const TensorShape& shape) const {
    return CopyToHostAsync(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode LlmHostDevice::Copy(BufferDesc* dst, const BufferDesc& src, uint64_t bytes) const {
    memcpy(dst->addr, src.addr, bytes);
    return RC_SUCCESS;
}

RetCode LlmHostDevice::Copy(BufferDesc* dst, const BufferDesc& src, const TensorShape& shape) const {
    return Copy(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode LlmHostDevice::ConvertToHostCommon(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                           const TensorShape& src_desc, const void* src_custom_info,
                                           const function<RetCode(void*, const BufferDesc&, const TensorShape&)>& copy_fn) {
    if (src_desc.CalcBytesExcludingPadding() == 0) {
        // TODO release dst
        return RC_SUCCESS;
    }

    if (dst_desc.GetDataFormat() == src_desc.GetDataFormat() && dst_desc.GetDataType() == src_desc.GetDataType()) {
        auto status = copy_fn(dst, src, src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy dst data to Host failed: " << GetRetCodeStr(status);
        }
        return status;
    }

    LOG(ERROR) << "only support same dataformat and datatype convert"
        << ", src: [" << GetDataTypeStr(src_desc.GetDataType()) << ", " << GetDataFormatStr(src_desc.GetDataFormat()) << "]"
        << ", dst: [" << GetDataTypeStr(dst_desc.GetDataType()) << ", " << GetDataFormatStr(dst_desc.GetDataFormat()) << "]";
    return RC_UNSUPPORTED;
}

RetCode LlmHostDevice::ConvertToHost(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                     const TensorShape& src_desc, const void* src_custom_info) {
    return ConvertToHostCommon(dst, dst_desc, src, src_desc, src_custom_info,
                               [this](void* dst, const BufferDesc& src, const TensorShape& src_desc) -> RetCode {
                                   return CopyToHost(dst, src, src_desc);
                               });
}

RetCode LlmHostDevice::ConvertToHostAsync(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                          const TensorShape& src_desc, const void* src_custom_info) {
    return ConvertToHostCommon(dst, dst_desc, src, src_desc, src_custom_info,
                               [this](void* dst, const BufferDesc& src, const TensorShape& src_desc) -> RetCode {
                                   return CopyToHostAsync(dst, src, src_desc);
                               });
}

RetCode LlmHostDevice::ConvertFromHostCommon(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                             const TensorShape& src_desc, const void* dst_custom_info,
                                             const function<RetCode(BufferDesc*, const void*, const TensorShape&)>& copy_fn) {
    if (src_desc.CalcBytesExcludingPadding() == 0) {
        Free(dst);
        return RC_SUCCESS;
    }

    if (dst_desc.GetDataFormat() == src_desc.GetDataFormat() && dst_desc.GetDataType() == src_desc.GetDataType()) {
        auto status = copy_fn(dst, src, src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy src data from host failed: " << GetRetCodeStr(status);
        }
        return status;
    }

    LOG(ERROR) << "only support same dataformat and datatype convert"
        << ", src: [" << GetDataTypeStr(src_desc.GetDataType()) << ", " << GetDataFormatStr(src_desc.GetDataFormat()) << "]"
        << ", dst: [" << GetDataTypeStr(dst_desc.GetDataType()) << ", " << GetDataFormatStr(dst_desc.GetDataFormat()) << "]";
    return RC_UNSUPPORTED;
}

RetCode LlmHostDevice::ConvertFromHost(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                       const TensorShape& src_desc, const void* dst_custom_info) {
    return ConvertFromHostCommon(dst, dst_desc, src, src_desc, dst_custom_info,
                                 [this](BufferDesc* dst, const void* src, const TensorShape& src_desc) -> RetCode {
                                     return CopyFromHost(dst, src, src_desc);
                                 });
}

RetCode LlmHostDevice::ConvertFromHostAsync(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                            const TensorShape& src_desc, const void* dst_custom_info) {
    return ConvertFromHostCommon(dst, dst_desc, src, src_desc, dst_custom_info,
                                 [this](BufferDesc* dst, const void* src, const TensorShape& src_desc) -> RetCode {
                                     return CopyFromHostAsync(dst, src, src_desc);
                                 });
}

RetCode LlmHostDevice::Convert(BufferDesc* dst, const TensorShape& dst_desc, const BufferDesc& src,
                               const TensorShape& src_desc, const void* dst_custom_info,
                               const void* src_custom_info) {
    LOG(ERROR) << "do not support custom convert";
    return RC_UNSUPPORTED;
}

/* -------------------------------------------------------------------------- */

// LlmHostDevice::ConfHandlerFunc LlmHostDevice::conf_handlers_[];

RetCode LlmHostDevice::Configure(uint32_t option, ...) {
    LOG(ERROR) << "configure is unsupported for LlmHostDevice";
    return RC_INVALID_VALUE;
}

}}}} // ppl::nn::llm::cuda
