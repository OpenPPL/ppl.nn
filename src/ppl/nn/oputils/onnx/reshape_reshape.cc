#include "ppl/nn/oputils/onnx/reshape_reshape.h"
#include "ppl/nn/common/logger.h"
#include <memory>
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeReshape(InputOutputInfo* info, const void*, const int64_t* shape_data) {
    auto data = &info->GetInput<TensorImpl>(0)->GetShape();
    auto shape = &info->GetInput<TensorImpl>(1)->GetShape();
    auto reshaped = &info->GetOutput<TensorImpl>(0)->GetShape();

    if (shape->GetDimCount() != 1) {
        LOG(ERROR) << "shape must be 1D tensor.";
        return RC_INVALID_VALUE;
    }

    reshaped->SetDimCount(shape->GetDim(0));
    int32_t axis_need_infer = -1;
    for (uint32_t i = 0; i < shape->GetDim(0); ++i) {
        if (shape_data[i] == -1) {
            if (axis_need_infer == -1) {
                axis_need_infer = i;
                reshaped->SetDim(i, 1);
            } else {
                LOG(ERROR) << "more than one axes need infer.";
                return RC_INVALID_VALUE;
            }
        } else if (shape_data[i] == 0) {
            if (i < data->GetDimCount()) {
                reshaped->SetDim(i, data->GetDim(i));
            } else {
                LOG(ERROR) << "axis to copy is greater than data dim count.";
                return RC_INVALID_VALUE;
            }
        } else {
            reshaped->SetDim(i, shape_data[i]);
        }
    }

    if (axis_need_infer != -1) {
        uint64_t data_nelem = data->GetElementsExcludingPadding();
        uint64_t pre_reshaped_nelem = reshaped->GetElementsExcludingPadding();
        if (pre_reshaped_nelem == 0) {
            LOG(ERROR) << "Reshaped tensor size is 0.";
            reshaped->SetDim(axis_need_infer, 0);
        } else if (data_nelem % pre_reshaped_nelem) {
            LOG(ERROR) << "infer shape failed.";
            return RC_INVALID_VALUE;
        } else {
            reshaped->SetDim(axis_need_infer, data_nelem / pre_reshaped_nelem);
        }
    }

    reshaped->CalcPadding();
    return RC_SUCCESS;
}

RetCode ReshapeReshape(InputOutputInfo* info, const void* arg) {
    if (info->GetInputCount() != 2) {
        LOG(ERROR) << "2 input required.";
        return RC_INVALID_VALUE;
    }

    auto shape_data = info->GetInput<TensorImpl>(1)->GetBufferPtr<int64_t>();
    if (!shape_data) {
        return RC_NOT_FOUND;
    }
    return ReshapeReshape(info, nullptr, shape_data);
}

}}} // namespace ppl::nn::oputils
