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

#include "ppl/nn/engines/cuda/default_cuda_allocator.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/buffered_cuda_device.h"
#include "ppl/nn/engines/cuda/buffered_cuda_allocator.h"
#include "ppl/nn/utils/stack_buffer_manager.h"
#include "ppl/nn/utils/compact_buffer_manager.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

#define DEFAULT_BLOCK_SIZE 1048576

RetCode BufferedCudaDevice::Init(const CudaEngineOptions& options) {
    CudaDevice::Init(options.device_id);

    if (options.mm_policy == CUDA_MM_BEST_FIT) {
        allocator_.reset(new DefaultCudaAllocator());
        buffer_manager_.reset(new utils::StackBufferManager(allocator_.get(), true));
    } else if (options.mm_policy == CUDA_MM_COMPACT) {
        size_t granularity = 0;
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = options.device_id;
        cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

        auto allocator = new BufferedCudaAllocator();
        auto status = allocator->Init(options.device_id, granularity);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "init BufferedCudaAllocator failed: " << GetRetCodeStr(status);
            delete allocator;
            return RC_OTHER_ERROR;
        }
        allocator_.reset(allocator);

        uint64_t block_size = DEFAULT_BLOCK_SIZE;
        if (granularity > DEFAULT_BLOCK_SIZE) {
            block_size = granularity;
        }
        buffer_manager_.reset(new utils::CompactBufferManager(allocator, block_size));
    }

    return RC_SUCCESS;
}

BufferedCudaDevice::~BufferedCudaDevice() {
    if (buffer_manager_.get()) {
        LOG(DEBUG) << "buffer manager[" << buffer_manager_->GetName() << "] allocates ["
                   << buffer_manager_->GetAllocatedBytes() << "] bytes.";
        buffer_manager_->Free(&shared_tmp_buffer_);
    }
    buffer_manager_.reset();
    allocator_.reset();
}

RetCode BufferedCudaDevice::Realloc(uint64_t bytes, BufferDesc* buffer) {
    return buffer_manager_->Realloc(bytes, buffer);
}

void BufferedCudaDevice::Free(BufferDesc* buffer) {
    if (buffer->addr) {
        buffer_manager_->Free(buffer);
    }
}

RetCode BufferedCudaDevice::AllocTmpBuffer(uint64_t bytes, BufferDesc* buffer) {
    if (bytes > tmp_buffer_size_) {
        auto status = buffer_manager_->Realloc(bytes, &shared_tmp_buffer_);
        if (status == RC_SUCCESS) {
            tmp_buffer_size_ = bytes;
            *buffer = shared_tmp_buffer_;
        }
        return status;
    }

    *buffer = shared_tmp_buffer_;
    return RC_SUCCESS;
}

void BufferedCudaDevice::FreeTmpBuffer(BufferDesc*) {}

}}} // namespace ppl::nn::cuda
