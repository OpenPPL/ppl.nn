#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_GEMM_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_GEMM_OP_H_

#include "ppl/nn/engines/cuda/params/gemm_extra_param.h"

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace cuda {

class GemmOp final : public CudaOptKernel {
public:
    GemmOp(const ir::Node* node) : CudaOptKernel(node) {}
    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode Init(const OptKernelOptions&) override;
    ppl::common::RetCode Finalize(const OptKernelOptions& options) override;
    void* GetParam() override {
        return (void*)&param_;
    };
    void CopyParam(void*& param) override;

private:
    CudaGemmParam param_;
};

}}} // namespace ppl::nn::cuda

#endif
