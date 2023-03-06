#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_PMX_LAYERNORM_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_PMX_LAYERNORM_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/pmx/layer_norm_param.h"

namespace ppl { namespace nn { namespace cuda {

class LayerNormKernel : public CudaKernel {
public:
    LayerNormKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::pmx::LayerNormParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::pmx::LayerNormParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
