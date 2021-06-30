#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_SPLIT_TO_SEQUENCE_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_SPLIT_TO_SEQUENCE_OP_H_

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

#include "ppl/nn/engines/common/onnx/split_to_sequence_op.h"

namespace ppl { namespace nn { namespace cuda {

class SplitToSequenceOp final : public CudaOptKernel {
public:
    SplitToSequenceOp(const ir::Node* node) : CudaOptKernel(node), op_(node) {}
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    KernelImpl* CreateKernelImpl() const override;

    ppl::common::RetCode Finalize(const OptKernelOptions&) override {
        return ppl::common::RC_SUCCESS;
    }

private:
    common::SplitToSequenceOp op_;
};

}}} // namespace ppl::nn::cuda

#endif
