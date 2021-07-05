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

#include "ppl/nn/oputils/onnx/reshape_pad.h"
#include "ppl/nn/params/onnx/pad_param.h"
using namespace std;
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapePad(InputOutputInfo* info, const void* arg, const int64_t* start_pads, const int64_t* end_pads) {
    auto param = (const PadParam*)arg;

    const TensorShape& shape = info->GetInput<TensorImpl>(0)->GetShape();
    int dim_count = shape.GetDimCount();
    int64_t output_dim[PAD_PARAM_MAX_DIM_SIZE];

    for (int it = 0; it < dim_count; ++it) {
        int start_pad = start_pads[it];
        int end_pad = end_pads[it];
        int cur_dim_size = shape.GetDim(it);

        if (start_pad < 0 || end_pad < 0) {
            return RC_INVALID_VALUE;
        }
        if (param->mode == PadParam::PAD_MODE_REFLECT && (start_pad >= cur_dim_size || end_pad >= cur_dim_size)) {
            return RC_INVALID_VALUE;
        }
        output_dim[it] = cur_dim_size + start_pad + end_pad;
    }
    info->GetOutput<TensorImpl>(0)->GetShape().Reshape(output_dim, dim_count);

    return RC_SUCCESS;
}

RetCode ReshapePad(InputOutputInfo* info, const void* arg) {
    const TensorShape& shape = info->GetInput<TensorImpl>(0)->GetShape();
    uint32_t dim_count = shape.GetDimCount();

    auto pad = info->GetInput<TensorImpl>(1);
    if (pad->GetShape().GetDimCount() != 1 || pad->GetShape().GetDim(0) != 2 * dim_count ||
        pad->GetShape().GetDataType() != DATATYPE_INT64) {
        return RC_INVALID_VALUE;
    }

    auto pads_data = pad->GetBufferPtr<int64_t>();
    if (!pads_data) {
        return RC_NOT_FOUND;
    }
    auto start_pads = pads_data;
    auto end_pads = pads_data + dim_count;

    return ReshapePad(info, arg, start_pads, end_pads);
}

}}} // namespace ppl::nn::oputils
