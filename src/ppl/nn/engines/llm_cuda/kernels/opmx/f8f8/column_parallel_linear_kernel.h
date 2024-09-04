#ifdef PPLNN_ENABLE_FP8

#ifndef _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_KERNELS_PMX_F8F8_COLUMN_PARALLEL_LINEAR_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_KERNELS_PMX_F8F8_COLUMN_PARALLEL_LINEAR_KERNEL_H_

#include "ppl/nn/engines/llm_cuda/kernel.h"
#include "ppl/nn/params/opmx/column_parallel_linear_param.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace opmx {

class F8F8ColumnParallelLinearKernel : public LlmCudaKernel {
public:
    F8F8ColumnParallelLinearKernel(const ir::Node* node) : LlmCudaKernel(node) {}

    void SetParam(const ppl::nn::opmx::ColumnParallelLinearParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::opmx::ColumnParallelLinearParam* param_ = nullptr;
};

}}}}} // namespace ppl::nn::llm::cuda::pmx

#endif
#endif
