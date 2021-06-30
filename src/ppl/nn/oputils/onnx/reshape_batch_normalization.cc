#include "ppl/nn/oputils/onnx/reshape_batch_normalization.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeBatchNormalization(InputOutputInfo* info, const void*) {
    if (info->GetInputCount() != 5 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }
    const TensorShape& in_shape0 = info->GetInput<TensorImpl>(0)->GetShape();
    auto out_shape0 = &info->GetOutput<TensorImpl>(0)->GetShape();
    out_shape0->Reshape(in_shape0.GetDims(), in_shape0.GetDimCount());
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
