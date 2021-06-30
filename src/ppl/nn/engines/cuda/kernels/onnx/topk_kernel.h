#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_TOPK_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_TOPK_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/onnx/topk_param.h"

namespace ppl { namespace nn { namespace cuda {

class TopKKernel : public CudaKernel {
public:
    TopKKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::common::TopKParam* p) {
        param_ = p;
    }

private:
    uint64_t CalcTmpBufferSize(const KernelExecContext&) const override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::TopKParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
