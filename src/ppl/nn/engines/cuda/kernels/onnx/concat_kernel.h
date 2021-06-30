#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_CONCAT_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_CONCAT_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/engines/cuda/params/concat_extra_param.h"

namespace ppl { namespace nn { namespace cuda {

class ConcatKernel : public CudaKernel {
public:
    ConcatKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const CudaConcatParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode BeforeExecute(KernelExecContext*) override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
    bool CanDoExecute(const KernelExecContext&) const override;

private:
    const CudaConcatParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
