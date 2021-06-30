#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_SEQUENCE_AT_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_SEQUENCE_AT_OP_H_

#include "ppl/nn/engines/common/onnx/sequence_at_op.h"

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace cuda {

class SequenceAtOp final : public CudaOptKernel {
public:
    SequenceAtOp(const ir::Node* node) : CudaOptKernel(node), op_(node) {}

    ppl::common::RetCode Init(const OptKernelOptions&) override {
        infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
            return type != ppl::common::DATATYPE_UNKNOWN ? InferDefaultType(info, type) : InferInheritedType(info);
        };

        infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
            auto in_shape = &info->GetInput<TensorImpl>(0)->GetShape();
            for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
                auto out_shape = &info->GetOutput<TensorImpl>(0)->GetShape();
                out_shape->Reshape(in_shape->GetDims(), in_shape->GetRealDimCount());
            }
            return ppl::common::RC_SUCCESS;
        };
        return ppl::common::RC_SUCCESS;
    }

    ppl::common::RetCode Finalize(const OptKernelOptions&) override {
        return ppl::common::RC_SUCCESS;
    }

    KernelImpl* CreateKernelImpl() const override {
        return op_.CreateKernelImpl();
    }

private:
    common::SequenceAtOp op_;
};

}}} // namespace ppl::nn::cuda

#endif
