#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_CONVTRANSPOSE_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_CONVTRANSPOSE_KERNEL_H_

#include "ppl/nn/engines/x86/kernel.h"
#include "ppl/nn/params/onnx/convtranspose_param.h"

namespace ppl { namespace nn { namespace x86 {

class ConvTransposeKernel : public X86Kernel {
public:
    ConvTransposeKernel(const ir::Node* node) : X86Kernel(node) {}

    void SetParam(const ppl::nn::common::ConvTransposeParam* p) {
        param_ = p;
    }

private:
    uint64_t CalcTmpBufferSize(const KernelExecContext& ctx) const override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::ConvTransposeParam* param_ = nullptr;
};

}}} // namespace ppl::nn::x86

#endif
