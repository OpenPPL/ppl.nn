#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_CONV_DEPTHWISE_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_CONV_DEPTHWISE_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "cudakernel/nn/conv_depthwise.h"
#include "ppl/nn/engines/cuda/params/conv_extra_param.h"

namespace ppl { namespace nn { namespace cuda {

class ConvDepthwiseKernel : public CudaKernel {
public:
    ConvDepthwiseKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const CudaConvParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode BeforeExecute(KernelExecContext*) override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const CudaConvParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
