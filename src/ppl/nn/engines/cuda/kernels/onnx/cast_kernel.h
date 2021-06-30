#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_CAST_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_CAST_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/onnx/cast_param.h"

namespace ppl { namespace nn { namespace cuda {

class CastKernel : public CudaKernel {
public:
    CastKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::common::CastParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::CastParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
