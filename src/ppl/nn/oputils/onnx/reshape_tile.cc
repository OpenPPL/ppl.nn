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

#include "ppl/nn/oputils/onnx/reshape_tile.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ReshapeTile(InputOutputInfo* info, const void*, const int64_t* repeats) {
    if (info->GetInputCount() != 2 || info->GetOutputCount() != 1) {
        LOG(DEBUG) << "ERROR: input count[" << info->GetInputCount() << "] != 2 or output count["
                   << info->GetOutputCount() << "] != 1.";
        return RC_INVALID_VALUE;
    }

    const TensorShape& in_shape = *info->GetInput<TensorImpl>(0)->GetShape();
    const TensorShape& repeats_shape = *info->GetInput<TensorImpl>(1)->GetShape();

    if (in_shape.GetDimCount() != repeats_shape.GetDim(0)) {
        LOG(DEBUG) << "ERROR: input[0]'s dim count[" << in_shape.GetDimCount() << "] != input[1]'s dim[0]'s value["
                   << repeats_shape.GetDim(0) << "].";
        return RC_INVALID_VALUE;
    }

    if (repeats_shape.GetDimCount() != 1) {
        LOG(DEBUG) << "ERROR: input[1]'s dim count[" << repeats_shape.GetDimCount() << "] != 1.";
        return RC_INVALID_VALUE;
    }
    if (repeats_shape.GetDataType() != DATATYPE_INT64) {
        LOG(DEBUG) << "ERROR: input[1]'s data type is not int64.";
        return RC_INVALID_VALUE;
    }

    TensorShape* out_shape = info->GetOutput<TensorImpl>(0)->GetShape();
    uint32_t input_dims = in_shape.GetDimCount();
    out_shape->SetDimCount(input_dims);

    for (uint32_t i = 0; i < input_dims; ++i) {
        auto out_dim = in_shape.GetDim(i) * repeats[i];
        out_shape->SetDim(i, out_dim);
    }
    out_shape->CalcPadding();
    return RC_SUCCESS;
}

RetCode ReshapeTile(InputOutputInfo* info, const void*) {
    if (info->GetInputCount() != 2 || info->GetOutputCount() != 1) {
        LOG(DEBUG) << "ERROR: input count[" << info->GetInputCount() << "] != 2 or output count["
                   << info->GetOutputCount() << "] != 1.";
        return RC_INVALID_VALUE;
    }

    const int64_t* repeats = info->GetInput<TensorImpl>(1)->GetBufferPtr<const int64_t>();
    if (!repeats) {
        LOG(DEBUG) << "ERROR: input[1] is empty.";
        return RC_NOT_FOUND;
    }

    return ReshapeTile(info, nullptr, repeats);
}

}}} // namespace ppl::nn::onnx
