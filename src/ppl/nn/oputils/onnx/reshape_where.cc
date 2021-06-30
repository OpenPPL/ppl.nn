#include "ppl/nn/oputils/onnx/reshape_where.h"
#include "ppl/common/log.h"
#include "ppl/nn/oputils/broadcast.h"
#include <algorithm>
#include <cmath>
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace oputils {

common::RetCode ReshapeWhere(InputOutputInfo* info, const void* arg) {
    if (info->GetInputCount() != 3) {
        LOG(ERROR) << "3 input required.";
        return RC_INVALID_VALUE;
    }

    auto out = info->GetOutput<TensorImpl>(0);

    MultiInputBroadCaster multi_input_bc;
    for (uint32_t i = 0; i < info->GetInputCount(); i++) {
        multi_input_bc.PushBackInputTensorShape(info->GetInput<TensorImpl>(i)->GetShape());
    }
    multi_input_bc.CalcBroadCast();
    if (!multi_input_bc.CanBroadCast()) {
        LOG(ERROR) << "unbroadcastable input.";
        return RC_INVALID_VALUE;
    }

    auto& output_shape = multi_input_bc.OutputTensorShape();
    if (output_shape.IsScalar()) {
        out->GetShape().ReshapeAsScalar();
    } else {
        out->GetShape().Reshape(output_shape.GetDims(), output_shape.GetDimCount());
    }
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
