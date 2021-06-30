#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_MUL_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_MUL_KERNEL_H_

#include "ppl/nn/engines/x86/kernel.h"

namespace ppl { namespace nn { namespace x86 {

class MulKernel : public X86Kernel {
public:
    MulKernel(const ir::Node* node) : X86Kernel(node) {}

    void SetFuseReLU(bool fuse_relu) {
        fuse_relu_ = fuse_relu;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
    bool fuse_relu_ = false;
};

}}} // namespace ppl::nn::x86

#endif
