#include "ppl/nn/oputils/onnx/reshape_flatten.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeFlatten(InputOutputInfo* info, const void* arg) {
    auto param = (const FlattenParam*)arg;
    if (info->GetInputCount() != 1 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    auto input = info->GetInput<TensorImpl>(0);
    auto output = info->GetOutput<TensorImpl>(0);

    const int32_t dim_count = input->GetShape().GetDimCount();
    if (param->axis < -dim_count || param->axis > dim_count) {
        return RC_INVALID_VALUE;
    }

    const int32_t axis = param->axis < 0 ? param->axis + dim_count : param->axis;

    int64_t outer_dim = 1;
    for (int32_t i = 0; i < axis; i++) {
        outer_dim *= input->GetShape().GetDim(i);
    }
    int64_t inner_dim = 1;
    for (int32_t i = axis; i < dim_count; i++) {
        inner_dim *= input->GetShape().GetDim(i);
    }

    output->GetShape().Reshape({outer_dim, inner_dim});
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
