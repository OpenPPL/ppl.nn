#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_FLOOR_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_FLOOR_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

namespace ppl { namespace nn { namespace cuda {

class FloorKernel : public CudaKernel {
public:
    FloorKernel(const ir::Node* node) : CudaKernel(node) {}

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
};

}}} // namespace ppl::nn::cuda

#endif
