#ifdef PPLNN_ENABLE_FP8

#ifndef _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_OPS_PMX_F8F8_ONLINE_CAST_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_OPS_PMX_F8F8_ONLINE_CAST_OP_H_

#include "ppl/nn/engines/llm_cuda/opt_kernel.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace opmx {

class F8F8OnlineCastOp final : public LlmCudaOptKernel {
public:
    F8F8OnlineCastOp(const ir::Node* node) : LlmCudaOptKernel(node) {}

    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode DoInit(const OptKernelOptions&) override;

private:
    ppl::common::RetCode CommonInit();
};

}}}}} // namespace ppl::nn::llm::cuda::pmx

#endif
#endif
