#include "ppl/nn/oputils/mmcv/reshape_mmcv_non_max_suppression.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeMMCVNonMaxSuppression(InputOutputInfo* info, const void*) {
    if (info->GetInputCount() != 2 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    const TensorShape& input_boxes = info->GetInput<TensorImpl>(0)->GetShape();
    const TensorShape& input_scores = info->GetInput<TensorImpl>(1)->GetShape();
    auto output = &info->GetOutput<TensorImpl>(0)->GetShape();

    if (input_boxes.GetDimCount() != 2 || input_scores.GetDimCount() != 1) {
        return RC_INVALID_VALUE;
    }

    if (input_boxes.GetDim(1) != 4 || input_boxes.GetDim(0) != input_scores.GetDim(0)) {
        return RC_INVALID_VALUE;
    }

    const int64_t num_boxes = input_boxes.GetDim(0);
    output->SetDataType(DATATYPE_INT64);
    output->Reshape({num_boxes}); // max output num
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
