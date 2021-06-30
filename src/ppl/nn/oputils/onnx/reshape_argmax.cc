#include "ppl/nn/oputils/onnx/reshape_argmax.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeArgMax(InputOutputInfo* info, const void* arg) {
    auto param = (const ArgMaxParam*)arg;
    const TensorShape& in_shape0 = info->GetInput<TensorImpl>(0)->GetShape();
    const uint32_t out_dim_count = in_shape0.GetDimCount();
    std::vector<int64_t> out_dims(out_dim_count);
    for (uint32_t i = 0; i < out_dim_count; ++i) {
        out_dims[i] = in_shape0.GetDim(i);
    }
    const uint32_t fixed_axis = param->axis >= 0 ? param->axis : param->axis + in_shape0.GetDimCount();
    if (param->keepdims) {
        out_dims[fixed_axis] = 1;
    } else {
        if (fixed_axis < out_dim_count)
            out_dims.erase(out_dims.begin() + fixed_axis);
    }
    info->GetOutput<TensorImpl>(0)->GetShape().Reshape(out_dims);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
