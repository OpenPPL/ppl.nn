#ifdef PPLNN_ENABLE_FP8

#ifndef _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_KERNELS_PMX_F8F8_ROW_PARALLEL_LINEAR_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_KERNELS_PMX_F8F8_ROW_PARALLEL_LINEAR_KERNEL_H_

#include "ppl/nn/engines/llm_cuda/kernel.h"
#include "ppl/nn/params/opmx/row_parallel_linear_param.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace opmx {

class F8F8RowParallelLinearKernel : public LlmCudaKernel {
public:
    F8F8RowParallelLinearKernel(const ir::Node* node) : LlmCudaKernel(node) {}

    void SetParam(const ppl::nn::opmx::RowParallelLinearParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::opmx::RowParallelLinearParam* param_ = nullptr;
};

}}}}} // namespace ppl::nn::llm::cuda

#endif
#endif
