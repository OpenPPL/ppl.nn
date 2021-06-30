#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_LEAKY_RELU_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_LEAKY_RELU_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/onnx/leaky_relu_param.h"

namespace ppl { namespace nn { namespace cuda {

class LeakyReluKernel : public CudaKernel {
public:
    LeakyReluKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::common::LeakyReLUParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::LeakyReLUParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
