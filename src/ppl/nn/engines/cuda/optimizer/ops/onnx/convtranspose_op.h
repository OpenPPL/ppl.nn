#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_CONVTRANSPOSE_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_CONVTRANSPOSE_OP_H_

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

#include "ppl/nn/params/onnx/convtranspose_param.h"

namespace ppl { namespace nn { namespace cuda {

class ConvTransposeOp final : public CudaOptKernel {
public:
    ConvTransposeOp(const ir::Node* node) : CudaOptKernel(node) {}
    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode Init(const OptKernelOptions&) override;
    ppl::common::RetCode Finalize(const OptKernelOptions& options) override;

private:
    ppl::nn::common::ConvTransposeParam param_;
};

}}} // namespace ppl::nn::cuda

#endif
