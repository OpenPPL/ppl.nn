#ifndef _ST_HPC_PPL_NN_ENGINES_COMMON_PPL_CONVERTER_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_COMMON_PPL_CONVERTER_KERNEL_H_

#include "ppl/nn/engines/common/common_kernel_impl.h"

namespace ppl { namespace nn { namespace common {

/**
   @class ConverterKernel
   @brief convert input[i] to output[i]. output[i]'s datatype is the same as input[i],
   dataformat is NDARRAY.
*/
class ConverterKernel final : public CommonKernelImpl {
public:
    ConverterKernel(const ir::Node* node) : CommonKernelImpl(node) {}

protected:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
};

}}} // namespace ppl::nn::common

#endif
