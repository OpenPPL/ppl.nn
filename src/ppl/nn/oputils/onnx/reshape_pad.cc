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
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

template <typename Tpad>
RetCode ReshapePad(InputOutputInfo* info, const ir::Attr* arg, const Tpad* start_pads, const Tpad* end_pads) {
    auto param = static_cast<const PadParam*>(arg);

    const TensorShape& shape = *info->GetInput<TensorImpl>(0)->GetShape();
    uint32_t dim_count = shape.GetDimCount();
    int64_t output_dim[PAD_PARAM_MAX_DIM_SIZE];

    for (uint32_t it = 0; it < dim_count; ++it) {
        Tpad start_pad = start_pads[it];
        Tpad end_pad = end_pads[it];
        int64_t cur_dim_size = shape.GetDim(it);

        if (start_pad < 0 || end_pad < 0) {
            LOG(DEBUG) << "ERROR: start pad[" << start_pad << "] < 0 or end pad[" << end_pad << "] < 0.";
            return RC_INVALID_VALUE;
        }
        if (param->mode == PadParam::PAD_MODE_REFLECT && (start_pad >= cur_dim_size || end_pad >= cur_dim_size)) {
            LOG(DEBUG) << "ERROR: PAD_MODE_REFLECT: start_pad[" << start_pad << "] >= dim[" << it << "]'s value["
                       << cur_dim_size << "] or end_pad[" << end_pad << "] >= dim[" << it << "]'s value["
                       << cur_dim_size << "].";
            return RC_INVALID_VALUE;
        }
        output_dim[it] = cur_dim_size + start_pad + end_pad;
    }
    info->GetOutput<TensorImpl>(0)->GetShape()->Reshape(output_dim, dim_count);

    return RC_SUCCESS;
}

RetCode ReshapePad(InputOutputInfo* info, const ir::Attr* arg) {
    const TensorShape& shape = *info->GetInput<TensorImpl>(0)->GetShape();
    uint32_t dim_count = shape.GetDimCount();

    auto pad = info->GetInput<TensorImpl>(1);
    auto pad_shape = pad->GetShape();
    if (pad_shape->GetDimCount() != 1) {
        LOG(DEBUG) << "ERROR: pad shape's dim count[" << pad_shape->GetDimCount() << "] != 1.";
        return RC_INVALID_VALUE;
    }
    if (pad_shape->GetDim(0) != 2 * dim_count) {
        LOG(DEBUG) << "ERROR: pad shape's dim[0]'s value[" << pad_shape->GetDim(0) << "] != 2 * dim_count[" << dim_count
                   << "].";
        return RC_INVALID_VALUE;
    }
    if (pad_shape->GetDataType() != DATATYPE_INT64) {
        LOG(DEBUG) << "ERROR: pad shape's data type[" << GetDataTypeStr(pad_shape->GetDataType()) << "] is not int64.";
        return RC_INVALID_VALUE;
    }

    auto pads_data = pad->GetBufferPtr<int64_t>();
    if (!pads_data) {
        LOG(DEBUG) << "ERROR: input[1]' pad data is empty.";
        return RC_NOT_FOUND;
    }
    auto start_pads = pads_data;
    auto end_pads = pads_data + dim_count;
    return ReshapePad(info, arg, start_pads, end_pads);
}

}}} // namespace ppl::nn::onnx
