#include "ppl/nn/oputils/onnx/reshape_scatter_elements.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeScatterElements(InputOutputInfo* info, const void* arg) {
    if (info->GetInputCount() != 3 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    const TensorShape& input_data = info->GetInput<TensorImpl>(0)->GetShape();
    const TensorShape& input_indices = info->GetInput<TensorImpl>(1)->GetShape();
    const TensorShape& input_updates = info->GetInput<TensorImpl>(2)->GetShape();
    auto output = &info->GetOutput<TensorImpl>(0)->GetShape();

    const uint32_t r = input_data.GetRealDimCount();
    const uint32_t q = input_indices.GetRealDimCount();
    const uint32_t u = input_updates.GetRealDimCount();
    if (r < 1 || q < 1) {
        return RC_INVALID_VALUE;
    }
    if (r != q || q != u) {
        return RC_INVALID_VALUE;
    }

    for (uint32_t i = 0; i < r; i++) {
        if (input_indices.GetDim(i) != input_updates.GetDim(i)) {
            return RC_INVALID_VALUE;
        }
    }

    output->Reshape(input_data.GetDims(), input_data.GetDimCount());
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
