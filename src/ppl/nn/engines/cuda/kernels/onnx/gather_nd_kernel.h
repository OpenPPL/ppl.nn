#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_GATHER_ND_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_GATHER_ND_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/onnx/gather_nd_param.h"

namespace ppl { namespace nn { namespace cuda {

class GatherNdKernel : public CudaKernel {
public:
    GatherNdKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::common::GatherNDParam* p) {
        param_ = p;
    }

private:
    uint64_t CalcTmpBufferSize(const KernelExecContext& ctx) const override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::GatherNDParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
