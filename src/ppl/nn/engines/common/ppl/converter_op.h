#ifndef _ST_HPC_PPL_NN_ENGINES_COMMON_PPL_CONVERTER_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_COMMON_PPL_CONVERTER_OP_H_

#include "ppl/nn/runtime/opt_kernel.h"
#include "ppl/nn/engines/common/ppl/converter_kernel.h"

namespace ppl { namespace nn { namespace common {

class ConverterOp final : public OptKernel {
public:
    ConverterOp(const ir::Node* node) : OptKernel(node) {}
    KernelImpl* CreateKernelImpl() const {
        return new ConverterKernel(GetNode());
    }
};

}}} // namespace ppl::nn::common

#endif
