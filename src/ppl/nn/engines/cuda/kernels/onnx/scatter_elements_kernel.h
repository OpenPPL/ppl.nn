#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_SCATTER_ELEMENTS_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_SCATTER_ELEMENTS_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/onnx/scatter_elements_param.h"

namespace ppl { namespace nn { namespace cuda {

class ScatterElementsKernel : public CudaKernel {
public:
    ScatterElementsKernel(const ir::Node* node) : CudaKernel(node) {}
    void SetParam(const ppl::nn::common::ScatterElementsParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
    bool CanDoExecute(const KernelExecContext&) const override;

private:
    const ppl::nn::common::ScatterElementsParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
