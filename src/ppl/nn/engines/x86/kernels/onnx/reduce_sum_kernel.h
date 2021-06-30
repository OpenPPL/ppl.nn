#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_REDUCE_SUM_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_REDUCE_SUM_KERNEL_H_

#include "ppl/nn/engines/x86/kernel.h"
#include "ppl/nn/params/onnx/reduce_param.h"

namespace ppl { namespace nn { namespace x86 {

class ReduceSumKernel : public X86Kernel {
public:
    ReduceSumKernel(const ir::Node* node) : X86Kernel(node) {}

    void SetParam(const ppl::nn::common::ReduceParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::ReduceParam* param_ = nullptr;
};

}}} // namespace ppl::nn::x86

#endif
