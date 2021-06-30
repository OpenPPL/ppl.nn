#ifndef _ST_HPC_PPL_NN_ENGINES_X86OPTIMIZER_OPS_PPL_REORDER_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_X86OPTIMIZER_OPS_PPL_REORDER_OP_H_

#include "ppl/nn/engines/x86/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace x86 {

class ReorderOp final : public X86OptKernel {
public:
    ReorderOp(const ir::Node* node) : X86OptKernel(node) {}
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    KernelImpl* CreateKernelImpl() const override;
};

}}} // namespace ppl::nn::x86

#endif
