#ifndef _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPS_ONNX_CONVTRANSPOSE_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPS_ONNX_CONVTRANSPOSE_OP_H_

#include "ppl/nn/params/onnx/convtranspose_param.h"
#include "ppl/nn/engines/x86/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace x86 {

class ConvTransposeOp final : public X86OptKernel {
public:
    ConvTransposeOp(const ir::Node* node) : X86OptKernel(node) {}
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    KernelImpl* CreateKernelImpl() const override;

private:
    std::shared_ptr<ppl::nn::common::ConvTransposeParam> param_;
};

}}} // namespace ppl::nn::x86

#endif
