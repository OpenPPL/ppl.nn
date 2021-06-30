#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_SOFTMAX_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_SOFTMAX_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/onnx/softmax_param.h"

namespace ppl { namespace nn { namespace cuda {

class SoftmaxKernel : public CudaKernel {
public:
    SoftmaxKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::common::SoftmaxParam* p) {
        param_ = p;
    }

private:
    uint64_t CalcTmpBufferSize(const KernelExecContext& ctx) const;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::SoftmaxParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
