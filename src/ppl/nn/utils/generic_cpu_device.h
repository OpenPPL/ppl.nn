#ifndef _ST_HPC_PPL_NN_UTILS_GENERIC_CPU_DEVICE_H_
#define _ST_HPC_PPL_NN_UTILS_GENERIC_CPU_DEVICE_H_

#include "ppl/nn/common/device.h"
#include "ppl/nn/utils/generic_cpu_data_converter.h"
#include "ppl/common/generic_cpu_allocator.h"

namespace ppl { namespace nn { namespace utils {

class GenericCpuDevice : public Device {
public:
    GenericCpuDevice(uint64_t alignment = 64) : allocator_(alignment) {}
    virtual ~GenericCpuDevice() {}

    /** @brief get the underlying allocator used to allocate/free memories */
    ppl::common::Allocator* GetAllocator() const {
        return &allocator_;
    }

    ppl::common::RetCode Realloc(uint64_t bytes, BufferDesc*) override;
    ppl::common::RetCode Realloc(const TensorShape&, BufferDesc*) override final;
    void Free(BufferDesc*) override;

    ppl::common::RetCode CopyFromHost(BufferDesc* dst, const void* src, uint64_t bytes) const override;
    ppl::common::RetCode CopyFromHost(BufferDesc* dst, const void* src, const TensorShape&) const override;
    ppl::common::RetCode CopyToHost(void* dst, const BufferDesc& src, uint64_t bytes) const override;
    ppl::common::RetCode CopyToHost(void* dst, const BufferDesc& src, const TensorShape&) const override;

    ppl::common::RetCode Copy(BufferDesc* dst, const BufferDesc& src, uint64_t bytes) const override;
    ppl::common::RetCode Copy(BufferDesc* dst, const BufferDesc& src, const TensorShape&) const override;

    const DataConverter* GetDataConverter() const override {
        return &data_converter_;
    }

private:
    mutable ppl::common::GenericCpuAllocator allocator_;
    GenericCpuDataConverter data_converter_;
};

}}} // namespace ppl::nn::utils

#endif
