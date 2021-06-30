#include "ppl/nn/oputils/onnx/reshape_reduce.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeReduce(InputOutputInfo* info, const void* arg) {
    auto param = (const ReduceParam*)arg;
    auto x = &info->GetInput<TensorImpl>(0)->GetShape();
    auto y = &info->GetOutput<TensorImpl>(0)->GetShape();

    // check & prepare axes
    if (param->axes.size() > x->GetDimCount()) {
        return RC_INVALID_VALUE;
    }

    const uint32_t dim_count = x->GetDimCount();
    auto fixed_axes = param->axes;
    if (param->axes.empty()) { // empty axes means reduce all dimss
        y->ReshapeAsScalar();
        return RC_SUCCESS;
    }

    for (uint32_t i = 0; i < fixed_axes.size(); i++) {
        if (fixed_axes[i] >= (int)dim_count || fixed_axes[i] < -(int)dim_count) {
            return RC_INVALID_VALUE;
        }
        if (fixed_axes[i] < 0) { // turn negative axes to positive axes
            fixed_axes[i] = fixed_axes[i] + dim_count;
        }
    }

    // reshape
    y->Reshape(x->GetDims(), x->GetDimCount());
    if (param->keep_dims) {
        for (uint32_t a = 0; a < fixed_axes.size(); ++a) {
            y->SetDim(fixed_axes[a], 1);
        }
    } else {
        for (uint32_t a = 0; a < fixed_axes.size(); ++a) {
            y->SetDim(fixed_axes[a] - a, 0);
            for (size_t i = fixed_axes[a] + 1; i < x->GetDimCount(); ++i) {
                y->SetDim(i - a - 1, x->GetDim(i));
            }
            y->SetDimCount(y->GetDimCount() - 1);
        }
        if (y->GetDimCount() == 0) {
            y->ReshapeAsScalar();
        }
    }
    y->CalcPadding();

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
