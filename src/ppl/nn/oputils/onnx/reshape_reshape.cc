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

#include "ppl/nn/oputils/onnx/reshape_reshape.h"
#include <memory>
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeReshape(InputOutputInfo* info, const void*, const int64_t* shape_data) {
    auto data = &info->GetInput<TensorImpl>(0)->GetShape();
    auto shape = &info->GetInput<TensorImpl>(1)->GetShape();
    auto reshaped = &info->GetOutput<TensorImpl>(0)->GetShape();

    if (shape->GetDimCount() != 1) {
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
                return RC_INVALID_VALUE;
            }
        } else if (shape_data[i] == 0) {
            if (i < data->GetDimCount()) {
                reshaped->SetDim(i, data->GetDim(i));
            } else {
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
            reshaped->SetDim(axis_need_infer, 0);
        } else if (data_nelem % pre_reshaped_nelem) {
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
        return RC_INVALID_VALUE;
    }

    auto shape_data = info->GetInput<TensorImpl>(1)->GetBufferPtr<int64_t>();
    if (!shape_data) {
        return RC_NOT_FOUND;
    }
    return ReshapeReshape(info, nullptr, shape_data);
}

}}} // namespace ppl::nn::oputils
