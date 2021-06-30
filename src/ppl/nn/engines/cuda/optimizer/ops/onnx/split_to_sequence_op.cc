#include "ppl/nn/engines/cuda/optimizer/ops/onnx/split_to_sequence_op.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

RetCode SplitToSequenceOp::Init(const OptKernelOptions& options) {
    auto node = GetNode();
    auto graph_data = options.graph->data.get();
    auto attr_ref = graph_data->attrs.find(node->GetId());
    if (attr_ref == graph_data->attrs.end()) {
        LOG(ERROR) << "cannot find attr for SplitToSequenceOp[" << node->GetName() << "]";
        return RC_NOT_FOUND;
    }

    auto param = static_cast<ppl::nn::onnx::SplitToSequenceParam*>(attr_ref->second.get());
    op_.Init(param->axis, param->keepdims, common::SplitToSequenceOp::GenericSplitFunc);

    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        return type != DATATYPE_UNKNOWN ? InferDefaultType(info, type) : InferInheritedType(info);
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto in_shape = &info->GetInput<TensorImpl>(0)->GetShape();
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto out_shape = &info->GetOutput<TensorImpl>(0)->GetShape();
            out_shape->Reshape(in_shape->GetDims(), in_shape->GetRealDimCount());
        }
        return RC_SUCCESS;
    };

    return RC_SUCCESS;
}

KernelImpl* SplitToSequenceOp::CreateKernelImpl() const {
    return op_.CreateKernelImpl();
}

}}} // namespace ppl::nn::cuda
