#include "ppl/nn/oputils/onnx/reshape_resize.h"
#include <vector>
using namespace std;
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeResize(InputOutputInfo* info, const void* arg, const float* roi_data, const float* scales_data,
                      const int64_t* sizes_data) {
    if (scales_data && sizes_data) {
        return RC_INVALID_VALUE;
    }

    auto param = (const ResizeParam*)arg;
    const TensorShape& in_shape = info->GetInput<TensorImpl>(0)->GetShape();

    uint32_t input_dim_count = in_shape.GetDimCount();
    std::vector<uint32_t> out_dims(input_dim_count);

    if (scales_data) {
        auto scales_shape = &info->GetInput<TensorImpl>(2)->GetShape();
        if (scales_shape->GetDimCount() != 1 || scales_shape->GetDim(0) != input_dim_count) {
            return RC_INVALID_VALUE;
        }

        if (param->coord_trans_mode == ResizeParam::RESIZE_COORD_TRANS_MODE_TF_CROP_AND_RESIZE) {
            TensorShape* roi_shape = &info->GetInput<TensorImpl>(1)->GetShape();
            if (roi_shape->GetDimCount() != 1 || roi_shape->GetDim(0) != input_dim_count * 2 || !roi_data) {
                return RC_INVALID_VALUE;
            }

            for (uint32_t i = 0; i < input_dim_count; ++i) {
                out_dims[i] = in_shape.GetDim(i) * (roi_data[i + input_dim_count] - roi_data[i]) * scales_data[i];
            }
        } else {
            for (uint32_t i = 0; i < input_dim_count; ++i) {
                out_dims[i] = in_shape.GetDim(i) * scales_data[i];
            }
        }
    } else {
        TensorShape* sizes_shape = &info->GetInput<TensorImpl>(3)->GetShape();
        if (sizes_shape->GetDimCount() != 1 || sizes_shape->GetDim(0) != input_dim_count) {
            return RC_INVALID_VALUE;
        }

        for (uint32_t i = 0; i < input_dim_count; ++i) {
            out_dims[i] = sizes_data[i];
        }
    }

    TensorShape* out_shape = &info->GetOutput<TensorImpl>(0)->GetShape();
    out_shape->SetDimCount(input_dim_count);
    for (uint32_t i = 0; i < input_dim_count; ++i) {
        out_shape->SetDim(i, out_dims[i]);
    }
    out_shape->CalcPadding();

    return RC_SUCCESS;
}

RetCode ReshapeResize(InputOutputInfo* info, const void* arg) {
    const float* roi_data = nullptr;
    if (!info->GetInput<TensorImpl>(1)->GetShape().IsEmpty()) {
        roi_data = info->GetInput<TensorImpl>(1)->GetBufferPtr<float>();
        if (roi_data == nullptr) {
            return RC_NOT_FOUND;
        }
    }

    const float* scales_data = nullptr;
    if (!info->GetInput<TensorImpl>(2)->GetShape().IsEmpty()) {
        scales_data = info->GetInput<TensorImpl>(2)->GetBufferPtr<float>();
        if (scales_data == nullptr) {
            return RC_NOT_FOUND;
        }
    }

    const int64_t* sizes_data = nullptr;
    if (info->GetInputCount() == 4) {
        if (!info->GetInput<TensorImpl>(3)->GetShape().IsEmpty()) {
            sizes_data = info->GetInput<TensorImpl>(3)->GetBufferPtr<int64_t>();
            if (sizes_data == nullptr) {
                return RC_NOT_FOUND;
            }
        }
    }

    return ReshapeResize(info, arg, roi_data, scales_data, sizes_data);
}

}}} // namespace ppl::nn::oputils
