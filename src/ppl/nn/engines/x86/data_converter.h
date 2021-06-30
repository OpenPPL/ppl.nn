#ifndef _ST_HPC_PPL_NN_ENGINES_X86_DATA_CONVERTER_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_DATA_CONVERTER_H_

#include "ppl/nn/common/data_converter.h"
#include "ppl/common/sys.h"

namespace ppl { namespace nn { namespace x86 {

class X86DataConverter final : public DataConverter {
public:
    X86DataConverter(ppl::common::isa_t isa) : isa_(isa) {}

    void SetISA(ppl::common::isa_t isa) {
        isa_ = isa;
    }
    const ppl::common::isa_t GetISA() const {
        return isa_;
    }

    ppl::common::RetCode ConvertToHost(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                       const TensorShape& src_desc) const override;

    ppl::common::RetCode ConvertFromHost(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                         const TensorShape& src_desc) const override;

    ppl::common::RetCode Convert(BufferDesc* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                 const TensorShape& src_desc) const override;

private:
    bool MayUseISA(uint32_t flag) const {
        return !!(isa_ & flag);
    }

    ppl::common::isa_t isa_;
};

}}} // namespace ppl::nn::x86

#endif
