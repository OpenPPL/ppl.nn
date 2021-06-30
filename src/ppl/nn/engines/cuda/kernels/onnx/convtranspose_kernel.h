#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_CONVTRANSPOSE_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_CONVTRANSPOSE_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/onnx/convtranspose_param.h"

namespace ppl { namespace nn { namespace cuda {

class ConvTransposeKernel : public CudaKernel {
public:
    ConvTransposeKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::common::ConvTransposeParam* p) {
        param_ = p;
    }

private:
    uint64_t CalcTmpBufferSize(const KernelExecContext&) const override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::ConvTransposeParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
