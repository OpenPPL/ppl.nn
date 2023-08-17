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

#include "ppl/nn/engines/cuda/plain_cuda_device.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

PlainCudaDevice::~PlainCudaDevice() {
    SyncStream();
}

RetCode PlainCudaDevice::Init(uint32_t device_id, ppl::common::NcclParam* tp_nccl_param) {
#if CUDA_VERSION < 11040
    LOG(ERROR) << "PlainCudaDevice is not supported when CUDA < 11.4.";
    return RC_UNSUPPORTED;
#else
    allocator_.Init(GetStream());
    return CudaDevice::Init(device_id, tp_nccl_param);
#endif
}

RetCode PlainCudaDevice::Realloc(uint64_t bytes, BufferDesc* buffer) {
#if CUDA_VERSION < 11040
    LOG(ERROR) << "PlainCudaDevice is not supported when CUDA < 11.4.";
    return RC_UNSUPPORTED;
#else
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
#endif
}

void PlainCudaDevice::Free(BufferDesc* buffer) {
#if CUDA_VERSION < 11040
    LOG(ERROR) << "PlainCudaDevice is not supported when CUDA < 11.4.";
#else
    if (buffer->addr) {
        allocator_->Free(buffer->addr);
        buffer->addr = nullptr;
        buffer->desc = 0;
    }
#endif
}

}}} // namespace ppl::nn::cuda
