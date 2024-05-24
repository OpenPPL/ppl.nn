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

#include "plain_device.h"

#include "ppl/common/cuda/cuda_plain_allocator.h"
#include "ppl/common/cuda/cuda_plain_async_allocator.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace llm { namespace cuda {

RetCode PlainDevice::DoInit() {
    if (async_) {
        auto ar = new CudaPlainAsyncAllocator();
        ar->Init(stream_);
        allocator_ = ar;
    } else {
        allocator_ = new CudaPlainAllocator();
    }
    return RC_SUCCESS;
}

PlainDevice::~PlainDevice() {
    DoDestroy();
}

void PlainDevice::DoDestroy() {
    delete allocator_;
}

RetCode PlainDevice::Realloc(uint64_t bytes, BufferDesc* buffer) {
    if (buffer->addr) {
        allocator_->Free(buffer->addr);
    }

    if (bytes == 0) {
        buffer->addr = nullptr;
        buffer->desc = 0;
        return RC_SUCCESS;
    }

    buffer->addr = allocator_->Alloc(bytes);
    if (!buffer->addr) {
        return RC_OUT_OF_MEMORY;
    }

    buffer->desc = bytes;
    return RC_SUCCESS;
}

void PlainDevice::Free(BufferDesc* buffer) {
    if (buffer->addr) {
        allocator_->Free(buffer->addr);
        buffer->addr = nullptr;
        buffer->desc = 0;
    }
}

}}}} // namespace ppl::nn::llm::cuda
