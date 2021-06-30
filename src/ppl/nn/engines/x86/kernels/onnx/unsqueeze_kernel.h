#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_UNSQUEEZE_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_UNSQUEEZE_KERNEL_H_

#include "ppl/nn/engines/x86/kernel.h"
#include "ppl/nn/params/onnx/unsqueeze_param.h"

namespace ppl { namespace nn { namespace x86 {

class UnsqueezeKernel : public X86Kernel {
public:
    UnsqueezeKernel(const ir::Node* node) : X86Kernel(node) {}

    void SetParam(const ppl::nn::common::UnsqueezeParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::UnsqueezeParam* param_ = nullptr;
};

}}} // namespace ppl::nn::x86

#endif
