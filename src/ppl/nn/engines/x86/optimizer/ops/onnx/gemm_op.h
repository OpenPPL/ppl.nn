#ifndef _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPS_ONNX_GEMM_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPS_ONNX_GEMM_OP_H_

#include "ppl/nn/params/onnx/gemm_param.h"
#include "ppl/nn/engines/x86/params/fc_param.h"
#include "ppl/nn/engines/x86/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace x86 {

class GemmOp final : public X86OptKernel {
public:
    GemmOp(const ir::Node* node) : X86OptKernel(node), fc_param_(nullptr) {}
    ~GemmOp();
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    KernelImpl* CreateKernelImpl() const override;
    bool SetFuseReLU();

private:
    FCParam* fc_param_;
    std::shared_ptr<ppl::nn::common::GemmParam> param_;
    bool gemm_fuse_relu_ = false;
};

}}} // namespace ppl::nn::x86

#endif
