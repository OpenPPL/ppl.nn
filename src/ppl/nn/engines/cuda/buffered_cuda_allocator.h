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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_BUFFERED_CUDA_ALLOCATOR_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_BUFFERED_CUDA_ALLOCATOR_H_

#include <cuda.h>
#include <vector>

#include "ppl/common/retcode.h"
#include "ppl/common/allocator.h"
#include "ppl/nn/engines/cuda/macros.h"

namespace ppl { namespace nn {

class BufferedCudaAllocator final : public ppl::common::Allocator {
public:
    BufferedCudaAllocator() : Allocator(CUDA_DEFAULT_ALIGNMENT) {}
    ~BufferedCudaAllocator();

    ppl::common::RetCode Init(int devid, uint64_t granularity);

    void* Alloc(uint64_t bytes) override;
    void Free(void* ptr) override;

private:
    ppl::common::RetCode InitCudaEnv(int device_id);

private:
    CUmemAllocationProp prop_ = {};
    CUmemAccessDesc access_desc_ = {};
    CUdeviceptr addr_ = 0;
    size_t addr_len_ = 0;
    size_t bytes_allocated_ = 0;
    size_t total_bytes_ = 0;
    std::vector<CUmemGenericAllocationHandle> handle_list_;

    CUcontext cu_context_;
    CUdevice cu_device_;

private:
    BufferedCudaAllocator(const BufferedCudaAllocator&) = delete;
    BufferedCudaAllocator& operator=(const BufferedCudaAllocator&) = delete;
};

}} // namespace ppl::nn

#endif
