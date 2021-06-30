#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_CONCAT_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_CONCAT_KERNEL_H_

#include "ppl/nn/engines/x86/kernel.h"
#include "ppl/nn/params/onnx/concat_param.h"

namespace ppl { namespace nn { namespace x86 {

class ConcatKernel : public X86Kernel {
public:
    ConcatKernel(const ir::Node* node) : X86Kernel(node) {}

    void SetParam(const ppl::nn::common::ConcatParam* p) {
        param_ = p;
    }

private:
    uint64_t CalcTmpBufferSize(const KernelExecContext& ctx) const override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
    bool CanDoExecute(const KernelExecContext&) const override;

private:
    const ppl::nn::common::ConcatParam* param_ = nullptr;
    std::vector<const void*> src_list_;
    std::vector<const TensorShape*> src_shape_list_;
};

}}} // namespace ppl::nn::x86

#endif
