#include "ppl/nn/utils/generic_cpu_device.h"
#include <cstring> // memcpy
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace utils {

RetCode GenericCpuDevice::Realloc(uint64_t bytes, BufferDesc* buffer) {
    if (buffer->addr) {
        allocator_.Free(buffer->addr);
    }

    if (bytes == 0) {
        buffer->addr = nullptr;
        return RC_SUCCESS;
    }

    buffer->addr = allocator_.Alloc(bytes);
    if (!buffer->addr) {
        return RC_OUT_OF_MEMORY;
    }

    buffer->desc = bytes;
    return RC_SUCCESS;
}

RetCode GenericCpuDevice::Realloc(const TensorShape& shape, BufferDesc* buffer) {
    return Realloc(shape.GetBytesIncludingPadding(), buffer);
}

void GenericCpuDevice::Free(BufferDesc* buffer) {
    if (buffer->addr) {
        allocator_.Free(buffer->addr);
        buffer->addr = nullptr;
    }
}

RetCode GenericCpuDevice::CopyFromHost(BufferDesc* dst, const void* src, uint64_t bytes) const {
    memcpy(dst->addr, src, bytes);
    return RC_SUCCESS;
}

RetCode GenericCpuDevice::CopyFromHost(BufferDesc* dst, const void* src, const TensorShape& shape) const {
    return CopyFromHost(dst, src, shape.GetBytesIncludingPadding());
}

RetCode GenericCpuDevice::CopyToHost(void* dst, const BufferDesc& src, uint64_t bytes) const {
    memcpy(dst, src.addr, bytes);
    return RC_SUCCESS;
}

RetCode GenericCpuDevice::CopyToHost(void* dst, const BufferDesc& src, const TensorShape& shape) const {
    return CopyToHost(dst, src, shape.GetBytesIncludingPadding());
}

RetCode GenericCpuDevice::Copy(BufferDesc* dst, const BufferDesc& src, uint64_t bytes) const {
    memcpy(dst->addr, src.addr, bytes);
    return RC_SUCCESS;
}

RetCode GenericCpuDevice::Copy(BufferDesc* dst, const BufferDesc& src, const TensorShape& shape) const {
    return Copy(dst, src, shape.GetBytesIncludingPadding());
}

}}} // namespace ppl::nn::utils
