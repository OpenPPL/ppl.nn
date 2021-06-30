#ifndef _ST_HPC_PPL_NN_ENGINES_X86_X86_DEVICE_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_X86_DEVICE_H_

#include "ppl/nn/utils/generic_cpu_device.h"
#include "ppl/nn/engines/x86/data_converter.h"

namespace ppl { namespace nn { namespace x86 {

class X86Device : public utils::GenericCpuDevice {
public:
    X86Device(uint64_t alignment, ppl::common::isa_t isa)
        : GenericCpuDevice(alignment), isa_(isa), data_converter_(isa) {}

    void SetISA(ppl::common::isa_t isa) {
        isa_ = isa;
        data_converter_.SetISA(isa);
    }
    const ppl::common::isa_t GetISA() const {
        return isa_;
    }

    virtual ppl::common::RetCode AllocTmpBuffer(uint64_t bytes, BufferDesc* buffer) {
        return Realloc(bytes, buffer);
    }
    virtual void FreeTmpBuffer(BufferDesc* buffer) {
        Free(buffer);
    }

    const DataConverter* GetDataConverter() const override final {
        return &data_converter_;
    }

private:
    ppl::common::isa_t isa_;
    X86DataConverter data_converter_;
};

}}} // namespace ppl::nn::x86

#endif
