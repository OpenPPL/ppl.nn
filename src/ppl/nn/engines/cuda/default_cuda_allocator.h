#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_DEFAULT_CUDA_ALLOCATOR_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_DEFAULT_CUDA_ALLOCATOR_H_

#include <cuda_runtime.h>

#include "ppl/common/allocator.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/macros.h"

namespace ppl { namespace nn {

class DefaultCudaAllocator : public ppl::common::Allocator {
public:
    DefaultCudaAllocator() : Allocator(CUDA_DEFAULT_ALIGNMENT) {}

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
