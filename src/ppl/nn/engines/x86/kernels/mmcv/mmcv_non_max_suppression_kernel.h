#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_MMCV_NON_MAX_SUPPRESSION_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_MMCV_NON_MAX_SUPPRESSION_KERNEL_H_

#include "ppl/nn/engines/x86/kernel.h"
#include "ppl/nn/params/mmcv/mmcv_non_max_suppression_param.h"

namespace ppl { namespace nn { namespace x86 {

class MMCVNonMaxSuppressionKernel : public X86Kernel {
public:
    MMCVNonMaxSuppressionKernel(const ir::Node* node) : X86Kernel(node) {}

    void SetParam(const ppl::nn::common::MMCVNMSParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::MMCVNMSParam* param_ = nullptr;
};

}}} // namespace ppl::nn::x86

#endif
