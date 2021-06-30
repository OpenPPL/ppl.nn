#include "ppl/nn/oputils/onnx/reshape_convolution.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/common/log.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeConvolution(InputOutputInfo* info, const void* arg) {
    auto param = (const ConvolutionParam*)arg;
    auto x = &info->GetInput<TensorImpl>(0)->GetShape();
    auto w = &info->GetInput<TensorImpl>(1)->GetShape();
    auto y = &info->GetOutput<TensorImpl>(0)->GetShape();
    auto num_output = w->GetDim(0);

    y->SetDimCount(x->GetDimCount());
    y->SetDim(0, x->GetDim(0));
    y->SetDim(1, num_output);

    const int32_t kernel_dims = x->GetDimCount() - 2;
    for (int32_t i = 0; i < kernel_dims; ++i) {
        const int32_t j = i + 2;
        const int32_t kernel_shape_eff = (w->GetDim(j) - 1) * param->dilations[i] + 1;
        const int32_t out_dim =
            ((int32_t)x->GetDim(j) + param->pads[i] + param->pads[i + kernel_dims] - kernel_shape_eff) /
                param->strides[i] +
            1;
        if (out_dim <= 0) {
            LOG(ERROR) << "Output Width or Height Is Invalid Value!";
            return RC_INVALID_VALUE;
        }
        y->SetDim(j, out_dim);
    }
    y->CalcPadding();

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
