#include "ppl/nn/oputils/onnx/reshape_gather_nd.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeGatherND(InputOutputInfo* info, const void*) {
    if (info->GetInputCount() != 2 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    auto input_data = &info->GetInput<TensorImpl>(0)->GetShape();
    auto input_indices = &info->GetInput<TensorImpl>(1)->GetShape();
    auto output = &info->GetOutput<TensorImpl>(0)->GetShape();

    const uint32_t r = input_data->GetRealDimCount();
    const uint32_t q = input_indices->GetRealDimCount();
    if (r < 1 || q < 1) {
        return RC_INVALID_VALUE;
    }
    const uint32_t last_indices_dim = input_indices->GetDim(q - 1);
    if (last_indices_dim < 1 || last_indices_dim > r) {
        return RC_INVALID_VALUE;
    }
    const uint32_t output_dim_count = q + r - last_indices_dim - 1;
    output->SetDimCount(output_dim_count);
    size_t i = 0;
    for (i = 0; i < q - 1; i++) {
        output->SetDim(i, input_indices->GetDim(i));
    }
    for (; i < output_dim_count; i++) {
        output->SetDim(i, input_data->GetDim(i - (q - 1) + last_indices_dim));
    }
    output->CalcPadding();
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
