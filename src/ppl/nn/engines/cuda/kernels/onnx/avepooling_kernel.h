#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_AVE_POOLING_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_AVE_POOLING_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/onnx/pooling_param.h"

namespace ppl { namespace nn { namespace cuda {

class AvePoolingKernel : public CudaKernel {
public:
    AvePoolingKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::common::PoolingParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::PoolingParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
