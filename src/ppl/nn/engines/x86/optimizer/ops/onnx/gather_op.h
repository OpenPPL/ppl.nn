#ifndef _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPS_ONNX_GATHER_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPS_ONNX_GATHER_OP_H_

#include "ppl/nn/params/onnx/gather_param.h"
#include "ppl/nn/engines/x86/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace x86 {

class GatherOp final : public X86OptKernel {
public:
    GatherOp(const ir::Node* node) : X86OptKernel(node) {}
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    KernelImpl* CreateKernelImpl() const override;

private:
    std::shared_ptr<ppl::nn::common::GatherParam> param_;
};

}}} // namespace ppl::nn::x86

#endif
