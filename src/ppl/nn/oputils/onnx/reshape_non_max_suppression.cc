#include "ppl/nn/oputils/onnx/reshape_non_max_suppression.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeNonMaxSuppression(InputOutputInfo* info, int64_t max_output_boxes_per_class) {
    if (info->GetInputCount() < 2 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    const TensorShape& input_boxes = info->GetInput<TensorImpl>(0)->GetShape();
    const TensorShape& input_scores = info->GetInput<TensorImpl>(1)->GetShape();
    auto output = &info->GetOutput<TensorImpl>(0)->GetShape();

    if (input_boxes.GetDimCount() != 3 || input_scores.GetDimCount() != 3) {
        return RC_INVALID_VALUE;
    }

    if (input_boxes.GetDim(2) != 4 || input_boxes.GetDim(0) != input_scores.GetDim(0) ||
        input_boxes.GetDim(1) != input_scores.GetDim(2)) {
        return RC_INVALID_VALUE;
    }

    const int64_t batch = input_scores.GetDim(0);
    const int64_t num_classes = input_scores.GetDim(1);
    const int64_t num_boxes = input_scores.GetDim(2);
    const int64_t num_max_output = std::min(max_output_boxes_per_class, num_boxes) * batch * num_classes;

    output->Reshape({num_max_output, 3});
    return RC_SUCCESS;
}

RetCode ReshapeNonMaxSuppression(InputOutputInfo* info) {
    int64_t max_output_boxes_per_class =
        info->GetInputCount() >= 3 ? (info->GetInput<TensorImpl>(2)->GetBufferPtr<int64_t>())[0] : 0;
    return ReshapeNonMaxSuppression(info, max_output_boxes_per_class);
}

}}} // namespace ppl::nn::oputils
