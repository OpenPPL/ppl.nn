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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_DEFAULT_CUDA_ALLOCATOR_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_DEFAULT_CUDA_ALLOCATOR_H_

#include <cuda_runtime.h>

#include "ppl/common/allocator.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/macros.h"

namespace ppl { namespace nn {

class DefaultCudaAllocator final : public ppl::common::Allocator {
public:
    void* Alloc(uint64_t size) override {
        void* ptr = nullptr;
        if (size > 0) {
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err != cudaSuccess) {
                LOG(ERROR) << "call cudaMalloc failed with error code: " << err << ", " << cudaGetErrorString(err)
                           << ", size is " << size;
                return nullptr;
            }
        }
        return ptr;
    }

    void Free(void* ptr) override {
        if (ptr != nullptr) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                LOG(ERROR) << "call cudaFree failed with error code: " << err << ", " << cudaGetErrorString(err);
            }
        }
    }
};

}} // namespace ppl::nn

#endif
