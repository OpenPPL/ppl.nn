#include "ppl/nn/oputils/onnx/reshape_roialign.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeROIAlign(InputOutputInfo* info, const void* arg) {
    if (info->GetInputCount() != 3 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    auto param = (const ROIAlignParam*)arg;
    auto x = &info->GetInput<TensorImpl>(0)->GetShape();
    auto rois = &info->GetInput<TensorImpl>(1)->GetShape();
    auto batch_indices = &info->GetInput<TensorImpl>(2)->GetShape();
    auto output = &info->GetOutput<TensorImpl>(0)->GetShape();

    if (x->GetDimCount() != 4) {
        return RC_INVALID_VALUE;
    }
    if (rois->GetDimCount() != 2 || rois->GetDim(1) != 4) {
        return RC_INVALID_VALUE;
    }
    if (batch_indices->GetDimCount() != 1) {
        return RC_INVALID_VALUE;
    }
    const uint32_t num_rois = rois->GetDim(0);
    const uint32_t channels = x->GetDim(1);
    if (batch_indices->GetDim(0) != num_rois) {
        return RC_INVALID_VALUE;
    }

    output->Reshape({num_rois, channels, param->output_height, param->output_width});
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
