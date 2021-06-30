#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_MAX_UNPOOL_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_MAX_UNPOOL_OP_H_

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

#include "ppl/nn/params/onnx/maxunpool_param.h"

namespace ppl { namespace nn { namespace cuda {

class MaxUnPoolOp final : public CudaOptKernel {
public:
    MaxUnPoolOp(const ir::Node* node) : CudaOptKernel(node) {}
    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode Init(const OptKernelOptions&) override;
    ppl::common::RetCode Finalize(const OptKernelOptions& options) override;

private:
    ppl::nn::common::MaxUnpoolParam param_;
};

}}} // namespace ppl::nn::cuda

#endif
