#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_SHAPE_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_SHAPE_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

namespace ppl { namespace nn { namespace cuda {

class ShapeKernel : public CudaKernel {
public:
    ShapeKernel(const ir::Node* node) : CudaKernel(node) {}

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
    bool CanDoExecute(const KernelExecContext&) const override;
};

}}} // namespace ppl::nn::cuda

#endif
