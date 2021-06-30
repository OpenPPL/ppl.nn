#ifndef _ST_HPC_PPL_NN_ENGINES_COMMON_ONNX_SEQUENCE_AT_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_COMMON_ONNX_SEQUENCE_AT_KERNEL_H_

#include "ppl/nn/engines/common/common_kernel_impl.h"

namespace ppl { namespace nn { namespace common {

class SequenceAtKernel final : public CommonKernelImpl {
public:
    SequenceAtKernel(const ir::Node* node) : CommonKernelImpl(node) {}

protected:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
};

}}} // namespace ppl::nn::common

#endif
