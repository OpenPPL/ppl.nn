#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_LEAKY_RELU_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_LEAKY_RELU_OP_H_

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

#include "ppl/nn/params/onnx/leaky_relu_param.h"

namespace ppl { namespace nn { namespace cuda {

class LeakyReluOp final : public CudaOptKernel {
public:
    LeakyReluOp(const ir::Node* node) : CudaOptKernel(node) {}
    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode Init(const OptKernelOptions&) override;
    ppl::common::RetCode Finalize(const OptKernelOptions& options) override;
    void* GetParam() override {
        return (void*)&param_;
    };
    void CopyParam(void*& param) override;

private:
    ppl::nn::common::LeakyReLUParam param_;
};

}}} // namespace ppl::nn::cuda

#endif
