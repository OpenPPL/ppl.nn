#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_LEAKY_RELU_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_LEAKY_RELU_KERNEL_H_

#include "ppl/nn/engines/x86/kernel.h"
#include "ppl/nn/params/onnx/leaky_relu_param.h"

namespace ppl { namespace nn { namespace x86 {

class LeakyReluKernel : public X86Kernel {
public:
    LeakyReluKernel(const ir::Node* node) : X86Kernel(node) {}

    void SetParam(const ppl::nn::common::LeakyReLUParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::LeakyReLUParam* param_ = nullptr;
};

}}} // namespace ppl::nn::x86

#endif
