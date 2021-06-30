#include "ppl/nn/oputils/onnx/reshape_gather.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeGather(InputOutputInfo* info, const void* arg) {
    auto param = (const GatherParam*)arg;

    if (info->GetInputCount() != 2 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    auto data = &info->GetInput<TensorImpl>(0)->GetShape();
    auto indices = &info->GetInput<TensorImpl>(1)->GetShape();
    auto output = &info->GetOutput<TensorImpl>(0)->GetShape();

    const int32_t r = data->GetRealDimCount();
    const int32_t q = indices->GetRealDimCount();
    if (r < 1) {
        return RC_INVALID_VALUE;
    }

    if (param->axis < -r || param->axis > r - 1) {
        return RC_INVALID_VALUE;
    }

    if (indices->IsScalar()) {
        output->SetDimCount(r - 1);
        int32_t axis = param->axis < 0 ? param->axis + r : param->axis;
        if (axis == 0) {
            for (int32_t i = 0; i < r - 1; ++i) {
                output->SetDim(i, data->GetDim(i + 1));
            }
        } else {
            for (int32_t i = 0; i < axis; i++) {
                output->SetDim(i, data->GetDim(i));
            }
            for (int32_t i = axis; i < r - 1; i++) {
                output->SetDim(i, data->GetDim(i + 1));
            }
        }
    } else {
        output->SetDimCount(r + q - 1);
        int32_t axis = param->axis < 0 ? param->axis + r : param->axis;
        for (int32_t i = 0; i < q - 1; i++) {
            output->SetDim(i, indices->GetDim(i));
        }
        for (int32_t i = 0; i < axis; i++) {
            output->SetDim(i + q - 1, data->GetDim(i));
        }
        output->SetDim(axis + q - 1, indices->GetDim(q - 1));
        for (int32_t i = axis + 1; i < r; i++) {
            output->SetDim(i + q - 1, data->GetDim(i));
        }
    }
    output->CalcPadding();
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
