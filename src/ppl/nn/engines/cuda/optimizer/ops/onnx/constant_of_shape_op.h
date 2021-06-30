#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_CONSTANT_OF_SHAPE_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_CONSTANT_OF_SHAPE_OP_H_

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

#include "ppl/nn/params/onnx/constant_of_shape_param.h"

namespace ppl { namespace nn { namespace cuda {

class ConstantOfShapeOp final : public CudaOptKernel {
public:
    ConstantOfShapeOp(const ir::Node* node) : CudaOptKernel(node) {}
    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode Init(const OptKernelOptions&) override;
    ppl::common::RetCode Finalize(const OptKernelOptions& options) override;

private:
    ppl::nn::common::ConstantOfShapeParam param_;
};

}}} // namespace ppl::nn::cuda

#endif
