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

static inline uint64_t Align(uint64_t x, uint64_t n) {
    return (x + n - 1) & (~(n - 1));
}

BufferedCudaAllocator::~BufferedCudaAllocator() {
    if (addr_) {
        cuMemUnmap(addr_, addr_len_);
        cuMemAddressFree(addr_, addr_len_);
    }

    for (auto x = handle_list_.begin(); x != handle_list_.end(); ++x) {
        cuMemRelease(*x);
    }

    cuDevicePrimaryCtxRelease(cu_device_);
}

RetCode BufferedCudaAllocator::InitCudaEnv(int device_id) {
    CUresult status;
    const char* errmsg = nullptr;

    cu_device_ = device_id;

    status = cuInit(0);
    if (status != CUDA_SUCCESS) {
        cuGetErrorString(status, &errmsg);
        LOG(ERROR) << "cuInit failed: " << errmsg;
        return RC_OTHER_ERROR;
    }

    status = cuDevicePrimaryCtxRetain(&cu_context_, device_id);
    if (status != CUDA_SUCCESS) {
        cuGetErrorString(status, &errmsg);
        LOG(ERROR) << "cuDevicePrimaryCtxRetain failed: " << errmsg;
        return RC_OTHER_ERROR;
    }

    status = cuCtxSetCurrent(cu_context_);
    if (status != CUDA_SUCCESS) {
        cuGetErrorString(status, &errmsg);
        LOG(ERROR) << "cuCtxSetCurrent failed: " << errmsg;
        return RC_OTHER_ERROR;
    }

    return RC_SUCCESS;
}

RetCode BufferedCudaAllocator::Init(int devid, uint64_t granularity) {
    auto status = InitCudaEnv(devid);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "InitCudaEnv failed: " << GetRetCodeStr(status);
        return status;
    }

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

}} // namespace ppl::nn
