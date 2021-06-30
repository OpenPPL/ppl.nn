#ifndef _ST_HPC_PPL_NN_ENGINES_X86OPTIMIZER_OPS_MMCV_MMCV_ROIALIGN_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_X86OPTIMIZER_OPS_MMCV_MMCV_ROIALIGN_OP_H_

#include "ppl/nn/params/mmcv/mmcv_roialign_param.h"
#include "ppl/nn/engines/x86/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace x86 {

class MMCVROIAlignOp final : public X86OptKernel {
public:
    MMCVROIAlignOp(const ir::Node* node) : X86OptKernel(node) {}
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode SelectFormat(const InputOutputInfo& info,
                                      std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                      std::vector<ppl::common::dataformat_t>* selected_output_formats) override;

private:
    std::shared_ptr<ppl::nn::common::MMCVROIAlignParam> param_;
};

}}} // namespace ppl::nn::x86

#endif
