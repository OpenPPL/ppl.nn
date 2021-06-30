#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_ARGMAX_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_ARGMAX_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/onnx/argmax_param.h"

namespace ppl { namespace nn { namespace cuda {

class ArgMaxKernel : public CudaKernel {
public:
    ArgMaxKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::common::ArgMaxParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::ArgMaxParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
