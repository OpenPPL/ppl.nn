#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_SHAPE_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_SHAPE_KERNEL_H_

#include "ppl/nn/engines/x86/kernel.h"

namespace ppl { namespace nn { namespace x86 {

class ShapeKernel : public X86Kernel {
public:
    ShapeKernel(const ir::Node* node) : X86Kernel(node) {}

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

    bool CanDoExecute(const KernelExecContext&) const override;
};

}}} // namespace ppl::nn::x86

#endif
