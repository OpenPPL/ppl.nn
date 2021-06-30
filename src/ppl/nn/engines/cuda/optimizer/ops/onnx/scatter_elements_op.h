#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_SCATTER_ELEMENTS_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_SCATTER_ELEMENTS_OP_H_

#include "ppl/nn/params/onnx/scatter_elements_param.h"

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace cuda {

class ScatterElementsOp final : public CudaOptKernel {
public:
    ScatterElementsOp(const ir::Node* node) : CudaOptKernel(node) {}
    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode Init(const OptKernelOptions&) override;
    ppl::common::RetCode Finalize(const OptKernelOptions& options) override;

private:
    ppl::nn::common::ScatterElementsParam param_;
};

}}} // namespace ppl::nn::cuda

#endif
