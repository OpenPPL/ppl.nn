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

#include "ppl/nn/engines/cuda/buffered_cuda_allocator.h"

#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

#if PPLNN_CUDACC_VER_MAJOR * 1000 + PPLNN_CUDACC_VER_MINOR * 10 >= 10020
static inline uint64_t Align(uint64_t x, uint64_t n) {
    return (x + n - 1) & (~(n - 1));
}

BufferedCudaAllocator::~BufferedCudaAllocator() {
    if (!addr_) {
        return;
    }

    CUresult rc;
    const char* errmsg = nullptr;

    if (!handle_list_.empty()) {
        rc = cuMemUnmap(addr_, bytes_allocated_);
        if (rc != CUDA_SUCCESS) {
            cuGetErrorString(rc, &errmsg);
            LOG(ERROR) << "cuMemUnmap failed: " << errmsg;
        }

        for (auto x = handle_list_.begin(); x != handle_list_.end(); ++x) {
            rc = cuMemRelease(*x);
            if (rc != CUDA_SUCCESS) {
                cuGetErrorString(rc, &errmsg);
                LOG(ERROR) << "cuMemRelease failed: " << errmsg;
            }
        }
    }

    rc = cuMemAddressFree(addr_, addr_len_);
    if (rc != CUDA_SUCCESS) {
        cuGetErrorString(rc, &errmsg);
        LOG(ERROR) << "cuMemAddressFree failed: " << errmsg;
    }
}

RetCode BufferedCudaAllocator::Init(int devid, uint64_t granularity) {
    const char* errmsg = nullptr;

    auto rc = cuMemGetInfo(nullptr, &total_bytes_);
    if (rc != CUDA_SUCCESS) {
        cuGetErrorString(rc, &errmsg);
        LOG(ERROR) << "cuMemGetInfo failed: " << errmsg;
        return RC_OTHER_ERROR;
    }

    prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop_.location.id = devid;

    access_desc_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc_.location.id = devid;
    access_desc_.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    addr_len_ = Align(total_bytes_, granularity);
    rc = cuMemAddressReserve(&addr_, addr_len_, 0, 0, 0);
    if (rc != CUDA_SUCCESS) {
        addr_ = 0;
        cuGetErrorString(rc, &errmsg);
        LOG(ERROR) << "cuMemAddressReserve failed: " << errmsg;
        return RC_OTHER_ERROR;
    }

    return RC_SUCCESS;
}

void* BufferedCudaAllocator::Alloc(uint64_t bytes) {
    const char* errmsg = nullptr;
    CUmemGenericAllocationHandle alloc_handle;
    if (bytes >= total_bytes_) {
        LOG(ERROR) << "bytes[" << bytes << "] is larger than max [" << total_bytes_ << "]";
        return nullptr;
    }
    auto rc = cuMemCreate(&alloc_handle, bytes, &prop_, 0);
    if (rc != CUDA_SUCCESS) {
        cuGetErrorString(rc, &errmsg);
        LOG(ERROR) << "cuMemCreate [" << bytes << "] bytes failed: " << errmsg;
        return nullptr;
    }

    auto start_addr = addr_ + bytes_allocated_;

    rc = cuMemMap(start_addr, bytes, 0, alloc_handle, 0);
    if (rc != CUDA_SUCCESS) {
        cuGetErrorString(rc, &errmsg);
        LOG(ERROR) << "cuMemMap [" << bytes << "] to addr [" << start_addr << "] failed: " << errmsg;
        return nullptr;
    }

    cuMemSetAccess(start_addr, bytes, &access_desc_, 1);

    handle_list_.push_back(alloc_handle);
    bytes_allocated_ += bytes;

    return (void*)start_addr;
}

void BufferedCudaAllocator::Free(void*) {
    // do nothing
}
#endif

}} // namespace ppl::nn
