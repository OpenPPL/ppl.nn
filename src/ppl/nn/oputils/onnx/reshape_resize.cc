// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

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
    const TensorShape& in_shape = info->GetInput<TensorImpl>(0)->GetShape();
    uint32_t input_dim_count = in_shape.GetDimCount();

    auto roi = info->GetInputCount() > 1 ? info->GetInput<TensorImpl>(1) : nullptr;
    const float* roi_data = nullptr;
    if (roi && !roi->GetShape().IsEmpty()) {
        roi_data = info->GetInput<TensorImpl>(1)->GetBufferPtr<float>();
        if (roi_data == nullptr) {
            return RC_NOT_FOUND;
        }
    }

    auto scales = info->GetInputCount() > 2 ? info->GetInput<TensorImpl>(2) : nullptr;
    auto sizes = info->GetInputCount() > 3 ? info->GetInput<TensorImpl>(3) : nullptr;

    auto has_size = sizes && sizes->GetShape().GetDimCount() == 1 && sizes->GetShape().GetDim(0) == input_dim_count;
    auto has_scales = scales && scales->GetShape().GetDimCount() == 1 && scales->GetShape().GetDim(0) == input_dim_count;

    if (has_scales && has_size) {
        return RC_INVALID_VALUE;
    }

    if (!has_scales && !has_size) {
        return RC_INVALID_VALUE;
    }

    const float* scales_data = nullptr;
    if (has_scales) {
        scales_data = info->GetInput<TensorImpl>(2)->GetBufferPtr<float>();
        if (scales_data == nullptr) {
            return RC_NOT_FOUND;
        }
    }

    const int64_t* sizes_data = nullptr;
    if (has_size) {
        sizes_data = info->GetInput<TensorImpl>(3)->GetBufferPtr<int64_t>();
        if (sizes_data == nullptr) {
            return RC_NOT_FOUND;
        }
    }

    return ReshapeResize(info, arg, roi_data, scales_data, sizes_data);
}

}}} // namespace ppl::nn::oputils
