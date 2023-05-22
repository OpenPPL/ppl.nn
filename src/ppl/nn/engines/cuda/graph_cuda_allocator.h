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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_GRAPH_CUDA_ALLOCATOR_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_GRAPH_CUDA_ALLOCATOR_H_

#include <cuda_runtime.h>

#include "ppl/common/allocator.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/macros.h"

namespace ppl { namespace nn {

class GraphCudaAllocator final : public ppl::common::Allocator {
public:
    GraphCudaAllocator(cudaStream_t stream) : stream_(stream) {}

    void* Alloc(uint64_t size) override {
        void* ptr = nullptr;
        if (size > 0) {
            cudaStreamCaptureStatus status;
            auto err = cudaStreamIsCapturing(stream_, &status);
            if (err != cudaSuccess) {
                LOG(ERROR) << "call stream is capturing failed with err code: " << int(err) << ", "
                           << cudaGetErrorString(err);
                return nullptr;
            }
#if PPLNN_CUDACC_VER_MAJOR * 1000 + PPLNN_CUDACC_VER_MINOR * 10 >= 11040
            if (status == cudaStreamCaptureStatusActive) {
                err = cudaMallocAsync(&ptr, size, stream_);
                if (err != cudaSuccess) {
                    LOG(WARNING) << "call cudaMallocAsync failed with error code: " << (int)err << ", "
                                 << cudaGetErrorString(err) << ", size is " << size;
                    return nullptr;
                }
                return ptr;
            } else {
                err = cudaMalloc(&ptr, size);
                if (err != cudaSuccess) {
                    LOG(ERROR) << "call cudaMalloc failed with error code: " << (int)err << ", "
                               << cudaGetErrorString(err) << ", size is " << size;
                    return nullptr;
                }
            }
#else
            LOG(ERROR) << "Duw to lower CUDA version, Graph mode is not supported,please update cuda to 11.4 or above.";
            return nullptr;
#endif
        }
        return ptr;
    }

    void Free(void* ptr) override {
        if (ptr != nullptr) {
            cudaStreamCaptureStatus status;
            auto err = cudaStreamIsCapturing(stream_, &status);
            if (err != cudaSuccess) {
                LOG(ERROR) << "call stream is capturing failed with err code: " << int(err) << ", "
                           << cudaGetErrorString(err);
                return;
            }
#if PPLNN_CUDACC_VER_MAJOR * 1000 + PPLNN_CUDACC_VER_MINOR * 10 >= 11040
            if (status == cudaStreamCaptureStatusActive) {
                err = cudaFreeAsync(ptr, stream_);
                if (err != cudaSuccess) {
                    LOG(WARNING) << "call cudaFreeAsync failed with error code: " << (int)err << ", "
                                 << cudaGetErrorString(err);
                    return;
                }
                return;
            } else {
                err = cudaFree(ptr);
                if (err != cudaSuccess) {
                    LOG(ERROR) << "call cudaFree failed with error code: " << (int)err << ", "
                               << cudaGetErrorString(err);
                }
            }
#else
            LOG(ERROR) << "Duw to lower CUDA version, Graph mode is not supported,please update cuda to 11.4 or above.";
            return;
#endif
        }
    }

private:
    cudaStream_t stream_;
};
}} // namespace ppl::nn

#endif
