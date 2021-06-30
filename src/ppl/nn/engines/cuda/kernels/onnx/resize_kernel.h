#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_RESIZE_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_RESIZE_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/onnx/resize_param.h"

namespace ppl { namespace nn { namespace cuda {

class ResizeKernel : public CudaKernel {
public:
    ResizeKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::common::ResizeParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
    bool CanDoExecute(const KernelExecContext&) const override;

private:
    const ppl::nn::common::ResizeParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
