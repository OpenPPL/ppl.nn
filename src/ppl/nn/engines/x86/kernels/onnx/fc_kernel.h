#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_FC_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_FC_KERNEL_H_

#include "ppl/nn/engines/x86/kernel.h"
#include "ppl/nn/engines/x86/params/fc_param.h"
#include "ppl/kernel/x86/fp32/fc.h"

namespace ppl { namespace nn { namespace x86 {

class FCKernel : public X86Kernel {
public:
    FCKernel(const ir::Node* node) : X86Kernel(node) {}
    ~FCKernel() {
        if (executor_)
            delete executor_;
    }

    void SetParam(const FCParam* p) {
        if (executor_)
            delete executor_;
        executor_ = p->mgr->gen_executor();
    }

private:
    uint64_t CalcTmpBufferSize(const KernelExecContext& ctx) const override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    ppl::kernel::x86::fc_fp32_executor* executor_ = nullptr;
};

}}} // namespace ppl::nn::x86

#endif
