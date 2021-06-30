#include "ppl/nn/oputils/onnx/reshape_expand.h"
#include "ppl/nn/oputils/broadcast.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeExpand(InputOutputInfo* info, const void*, const int64_t* shape_ptr) {
    auto shape = info->GetInput<TensorImpl>(1);
    const uint32_t shape_dim_count = shape->GetShape().IsScalar() ? 1 : shape->GetShape().GetDim(0);

    const TensorShape& in_shape0 = info->GetInput<TensorImpl>(0)->GetShape();
    auto out_shape0 = &info->GetOutput<TensorImpl>(0)->GetShape();

    TensorShape expand_shape;
    expand_shape.Reshape(shape_ptr, shape_dim_count);

    MultiDirectionalBroadCaster broad_caster;
    broad_caster.SetInputTensorShapes(in_shape0, expand_shape);
    if (broad_caster.CanBroadCast() == false) {
        return RC_INVALID_VALUE;
    }

    auto& output_shape = broad_caster.OutputTensorShape();
    if (output_shape.IsScalar()) {
        out_shape0->ReshapeAsScalar();
    } else {
        out_shape0->Reshape(output_shape.GetDims(), output_shape.GetDimCount());
    }
    return RC_SUCCESS;
}

RetCode ReshapeExpand(InputOutputInfo* info, const void*) {
    if (info->GetInputCount() != 2 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    auto shape_ptr = info->GetInput<TensorImpl>(1)->GetBufferPtr<int64_t>();
    if (shape_ptr == nullptr) {
        return RC_NOT_FOUND;
    }
    return ReshapeExpand(info, nullptr, shape_ptr);
}

}}} // namespace ppl::nn::oputils
