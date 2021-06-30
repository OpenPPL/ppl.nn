#include "ppl/nn/oputils/onnx/reshape_pooling.h"
#include "ppl/common/log.h"
#include <cmath>
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapePooling(InputOutputInfo* info, const void* arg) {
    auto param = (const PoolingParam*)arg;
    auto x = &info->GetInput<TensorImpl>(0)->GetShape();
    auto y = &info->GetOutput<TensorImpl>(0)->GetShape();

    y->SetDimCount(x->GetDimCount());
    y->SetDim(0, x->GetDim(0));
    y->SetDim(1, x->GetDim(1));
    const int32_t kernel_dims = x->GetDimCount() - 2;
    if (param->global_pooling) {
        for (int32_t i = 2; i < kernel_dims + 2; ++i) {
            y->SetDim(i, std::min(1l, x->GetDim(i))); // input tensor dim may be zero
        }
    } else {
        std::vector<int32_t> pads(kernel_dims * 2, 0);
        for (uint32_t i = 0; i < param->pads.size(); i++) {
            pads[i] = param->pads[i];
        }
        std::vector<int32_t> dilations(kernel_dims, 1);
        for (uint32_t i = 0; i < param->dilations.size(); i++) {
            dilations[i] = param->dilations[i];
        }

        for (int32_t i = 0; i < kernel_dims; ++i) {
            const int32_t j = i + 2;
            const float pre_out_dim_f32 = (float)((int64_t)x->GetDim(j) + pads[i] + pads[i + kernel_dims] -
                                                  ((param->kernel_shape[i] - 1) * dilations[i] + 1)) /
                    param->strides[i] +
                1;
            int32_t out_dim_i32;
            if (param->ceil_mode) {
                out_dim_i32 = (int32_t)ceilf(pre_out_dim_f32);
            } else {
                out_dim_i32 = (int32_t)floorf(pre_out_dim_f32);
            }
            if (out_dim_i32 <= 0) {
                LOG(ERROR) << "Output Width or Height Is Invalid Value!";
                return RC_INVALID_VALUE;
            }
            y->SetDim(j, out_dim_i32);
        }
    }
    y->CalcPadding();

    if (info->GetOutputCount() == 2) {
        auto z = &info->GetOutput<TensorImpl>(1)->GetShape();
        z->SetDataType(DATATYPE_INT64);
        z->Reshape(y->GetDims(), y->GetDimCount());
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
