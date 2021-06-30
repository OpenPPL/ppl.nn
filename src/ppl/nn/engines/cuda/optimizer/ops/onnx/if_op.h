#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_IF_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_IF_OP_H_

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

#include "ppl/nn/engines/common/onnx/if_op.h"

namespace ppl { namespace nn { namespace cuda {

class IfOp final : public CudaOptKernel {
public:
    IfOp(const ir::Node* node) : CudaOptKernel(node), op_(node) {}

    ppl::common::RetCode Init(const OptKernelOptions& options) override;

    ppl::common::RetCode Finalize(const OptKernelOptions&) override {
        return ppl::common::RC_SUCCESS;
    }

    KernelImpl* CreateKernelImpl() const override;

private:
    common::IfOp op_;
};

}}} // namespace ppl::nn::cuda

#endif
