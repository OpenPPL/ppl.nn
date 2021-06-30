#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_ARGMAX_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_ARGMAX_KERNEL_H_

#include "ppl/nn/engines/x86/kernel.h"
#include "ppl/nn/params/onnx/argmax_param.h"

namespace ppl { namespace nn { namespace x86 {

class ArgMaxKernel : public X86Kernel {
public:
    ArgMaxKernel(const ir::Node* node) : X86Kernel(node) {}

    void SetParam(const ppl::nn::common::ArgMaxParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::ArgMaxParam* param_ = nullptr;
};

}}} // namespace ppl::nn::x86

#endif
