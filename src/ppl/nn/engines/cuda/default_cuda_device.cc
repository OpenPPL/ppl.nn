#include "ppl/nn/engines/cuda/default_cuda_device.h"

#include "ppl/nn/engines/cuda/default_cuda_allocator.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

DefaultCudaDevice::DefaultCudaDevice() {
    allocator_.reset(new DefaultCudaAllocator());
}

DefaultCudaDevice::~DefaultCudaDevice() {
    allocator_.reset();
}

RetCode DefaultCudaDevice::Realloc(uint64_t bytes, BufferDesc* buffer) {
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

void DefaultCudaDevice::Free(BufferDesc* buffer) {
    if (buffer->addr) {
        allocator_->Free(buffer->addr);
        buffer->addr = nullptr;
        buffer->desc = 0;
    }
}

}}} // namespace ppl::nn::cuda
