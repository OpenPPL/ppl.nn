#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_PMX_RMSNORM_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_PMX_RMSNORM_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/pmx/rms_norm_param.h"

namespace ppl { namespace nn { namespace cuda {

class RMSNormKernel : public CudaKernel {
public:
    RMSNormKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::pmx::RMSNormParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::pmx::RMSNormParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
