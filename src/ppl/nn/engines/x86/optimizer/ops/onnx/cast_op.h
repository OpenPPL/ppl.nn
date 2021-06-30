#ifndef _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPS_ONNX_CAST_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPS_ONNX_CAST_OP_H_

#include "ppl/nn/params/onnx/cast_param.h"
#include "ppl/nn/engines/x86/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace x86 {

class CastOp final : public X86OptKernel {
public:
    CastOp(const ir::Node* node) : X86OptKernel(node) {}
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    KernelImpl* CreateKernelImpl() const override;

private:
    std::shared_ptr<ppl::nn::common::CastParam> param_;
};

}}} // namespace ppl::nn::x86

#endif
