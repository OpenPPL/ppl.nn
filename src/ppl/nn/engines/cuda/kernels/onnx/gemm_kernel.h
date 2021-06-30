#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_GEMM_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_GEMM_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"
#include "ppl/nn/engines/cuda/params/gemm_extra_param.h"

namespace ppl { namespace nn { namespace cuda {

class GemmKernel : public CudaKernel {
public:
    GemmKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const CudaGemmParam* p) {
        param_ = p;
    }

private:
    uint64_t CalcTmpBufferSize(const KernelExecContext&) const override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const CudaGemmParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
