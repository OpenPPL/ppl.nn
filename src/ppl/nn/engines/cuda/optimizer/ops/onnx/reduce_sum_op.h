#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_REDUCE_SUM_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_REDUCE_SUM_OP_H_

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

#include "ppl/nn/params/onnx/reduce_param.h"

namespace ppl { namespace nn { namespace cuda {

class ReduceSumOp final : public CudaOptKernel {
public:
    ReduceSumOp(const ir::Node* node) : CudaOptKernel(node) {}
    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode Init(const OptKernelOptions&) override;
    ppl::common::RetCode Finalize(const OptKernelOptions& options) override;

private:
    ppl::nn::common::ReduceParam param_;
};

}}} // namespace ppl::nn::cuda

#endif
