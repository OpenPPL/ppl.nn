#ifndef _ST_HPC_PPL_NN_ENGINES_X86OPTIMIZER_OPS_MMCV_MMCV_NON_MAX_SUPPRESSION_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_X86OPTIMIZER_OPS_MMCV_MMCV_NON_MAX_SUPPRESSION_OP_H_

#include "ppl/nn/params/mmcv/mmcv_non_max_suppression_param.h"
#include "ppl/nn/engines/x86/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace x86 {

class MMCVNonMaxSuppressionOp final : public X86OptKernel {
public:
    MMCVNonMaxSuppressionOp(const ir::Node* node) : X86OptKernel(node) {}
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    KernelImpl* CreateKernelImpl() const override;

private:
    std::shared_ptr<ppl::nn::common::MMCVNMSParam> param_;
};

}}} // namespace ppl::nn::x86

#endif
