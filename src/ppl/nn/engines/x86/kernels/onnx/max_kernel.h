#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_MAX_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_MAX_KERNEL_H_

#include "ppl/nn/engines/x86/kernel.h"

namespace ppl { namespace nn { namespace x86 {

class MaxKernel : public X86Kernel {
public:
    MaxKernel(const ir::Node* node) : X86Kernel(node) {}

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
    uint64_t CalcTmpBufferSize(const KernelExecContext& ctx) const override;
};

}}} // namespace ppl::nn::x86

#endif
