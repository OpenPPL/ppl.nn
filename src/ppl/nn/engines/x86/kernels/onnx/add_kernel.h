#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_ADD_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_ADD_KERNEL_H_

#include "ppl/nn/engines/x86/kernel.h"

namespace ppl { namespace nn { namespace x86 {

class AddKernel : public X86Kernel {
public:
    AddKernel(const ir::Node* node) : X86Kernel(node) {}

    void SetFuseReLU(bool fuse_relu) {
        fuse_relu_ = fuse_relu;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
    bool fuse_relu_ = false;
};

}}} // namespace ppl::nn::x86

#endif
