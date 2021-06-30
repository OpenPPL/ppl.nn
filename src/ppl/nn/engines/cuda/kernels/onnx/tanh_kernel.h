#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_TANH_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_TANH_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

namespace ppl { namespace nn { namespace cuda {

class TanhKernel : public CudaKernel {
public:
    TanhKernel(const ir::Node* node) : CudaKernel(node) {}

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
};

}}} // namespace ppl::nn::cuda

#endif
