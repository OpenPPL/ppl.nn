#ifndef _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPS_ONNX_SEQUENCE_AT_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPS_ONNX_SEQUENCE_AT_OP_H_

#include "ppl/nn/engines/x86/optimizer/opt_kernel.h"
#include "ppl/nn/engines/common/onnx/sequence_at_op.h"

namespace ppl { namespace nn { namespace x86 {

class SequenceAtOp final : public X86OptKernel {
public:
    SequenceAtOp(const ir::Node* node) : X86OptKernel(node), op_(node) {}

    ppl::common::RetCode Init(const OptKernelOptions&) override {
        return ppl::common::RC_SUCCESS;
    }

    KernelImpl* CreateKernelImpl() const override {
        return op_.CreateKernelImpl();
    }

private:
    common::SequenceAtOp op_;
};

}}} // namespace ppl::nn::x86

#endif
