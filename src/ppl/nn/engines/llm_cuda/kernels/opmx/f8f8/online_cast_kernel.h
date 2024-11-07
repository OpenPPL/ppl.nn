#ifdef PPLNN_ENABLE_FP8

#ifndef _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_KERNELS_PMX_F8F8_ONLINE_CAST_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_KERNELS_PMX_F8F8_ONLINE_CAST_KERNEL_H_

#include "ppl/nn/engines/llm_cuda/kernel.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace opmx {

class F8F8OnlineCastKernel : public LlmCudaKernel {
public:
    F8F8OnlineCastKernel(const ir::Node* node) : LlmCudaKernel(node) {}

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

};

}}}}} // namespace ppl::nn::llm::cuda::pmx

#endif
#endif