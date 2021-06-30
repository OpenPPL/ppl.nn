#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_CAST_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_CAST_OP_H_

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

#include "ppl/nn/params/onnx/cast_param.h"

namespace ppl { namespace nn { namespace cuda {

class CastOp final : public CudaOptKernel {
public:
    CastOp(const ir::Node* node) : CudaOptKernel(node) {}
    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode Init(const OptKernelOptions&) override;
    ppl::common::RetCode Finalize(const OptKernelOptions& options) override;

    void* GetParam() override {
        return (void*)&param_;
    };
    bool CompareParam(CudaOptKernel* other) override {
        if (other->GetNode()->GetType().name == "Cast")
            return param_.to == ((ppl::nn::common::CastParam*)other->GetParam())->to;
        return false;
    }

private:
    ppl::nn::common::CastParam param_;
};

}}} // namespace ppl::nn::cuda

#endif
