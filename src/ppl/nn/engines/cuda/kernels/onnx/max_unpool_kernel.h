#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_MAX_UNPOOL_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_MAX_UNPOOL_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/onnx/maxunpool_param.h"

namespace ppl { namespace nn { namespace cuda {

class MaxUnpoolKernel : public CudaKernel {
public:
    MaxUnpoolKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::common::MaxUnpoolParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::MaxUnpoolParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
