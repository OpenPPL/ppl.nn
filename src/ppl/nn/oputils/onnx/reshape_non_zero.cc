#include "ppl/nn/oputils/onnx/reshape_non_zero.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeNonZero(InputOutputInfo* info, const void*) {
    if (info->GetInputCount() != 1 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    auto x = &info->GetInput<TensorImpl>(0)->GetShape();
    auto y = &info->GetOutput<TensorImpl>(0)->GetShape();

    const uint32_t input_dim_count = x->GetDimCount();
    const uint32_t max_output_num = x->GetElementsExcludingPadding();
    y->Reshape({input_dim_count, max_output_num});

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
