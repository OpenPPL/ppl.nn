#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_DEPTH_TO_SPACE_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_DEPTH_TO_SPACE_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/onnx/depth_to_space_param.h"

namespace ppl { namespace nn { namespace cuda {

class DepthToSpaceKernel : public CudaKernel {
public:
    DepthToSpaceKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::common::DepthToSpaceParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::DepthToSpaceParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
