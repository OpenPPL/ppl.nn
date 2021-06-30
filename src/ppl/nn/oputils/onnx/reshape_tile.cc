#include "ppl/nn/oputils/onnx/reshape_tile.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeTile(InputOutputInfo* info, const void* arg) {
    if (info->GetInputCount() != 2 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }
    if (info->GetInput<TensorImpl>(1)->GetShape().GetDataType() != DATATYPE_INT64) {
        return RC_INVALID_VALUE;
    }

    const TensorShape& in_shape = info->GetInput<TensorImpl>(0)->GetShape();
    const TensorShape& repeats_shape = info->GetInput<TensorImpl>(1)->GetShape();
    if (repeats_shape.GetDimCount() != 1 || in_shape.GetDimCount() != repeats_shape.GetDim(0)) {
        return RC_INVALID_VALUE;
    }

    const int64_t* repeats = info->GetInput<TensorImpl>(1)->GetBufferPtr<const int64_t>();
    if (repeats == nullptr) {
        return RC_NOT_FOUND;
    }

    uint32_t input_dims = info->GetInput<TensorImpl>(0)->GetShape().GetDimCount();
    info->GetOutput<TensorImpl>(0)->GetShape().SetDimCount(input_dims);

    uint32_t out_dims[input_dims];
    for (uint32_t i = 0; i < input_dims; ++i) {
        out_dims[i] = in_shape.GetDim(i) * repeats[i];
        info->GetOutput<TensorImpl>(0)->GetShape().SetDim(i, out_dims[i]);
    }
    info->GetOutput<TensorImpl>(0)->GetShape().CalcPadding();
    return RC_SUCCESS;
}

RetCode ReshapeTile(InputOutputInfo* info, const void* arg, const int64_t* repeats) {
    if (info->GetInputCount() != 2 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }
    if (info->GetInput<TensorImpl>(1)->GetShape().GetDataType() != DATATYPE_INT64) {
        return RC_INVALID_VALUE;
    }
    const TensorShape& in_shape = info->GetInput<TensorImpl>(0)->GetShape();

    uint32_t input_dims = info->GetInput<TensorImpl>(0)->GetShape().GetDimCount();
    info->GetOutput<TensorImpl>(0)->GetShape().SetDimCount(input_dims);

    uint32_t out_dims[input_dims];
    for (uint32_t i = 0; i < input_dims; ++i) {
        out_dims[i] = in_shape.GetDim(i) * repeats[i];
        info->GetOutput<TensorImpl>(0)->GetShape().SetDim(i, out_dims[i]);
    }
    info->GetOutput<TensorImpl>(0)->GetShape().CalcPadding();
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
