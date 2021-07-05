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

#include "ppl/nn/oputils/onnx/reshape_constant_of_shape.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeConstantOfShape(InputOutputInfo* info, const void* arg, const int64_t* input_host) {
    auto input = info->GetInput<TensorImpl>(0);
    auto input_shape = &input->GetShape();
    auto output_shape = &info->GetOutput<TensorImpl>(0)->GetShape();

    if (input_shape->GetDataType() != DATATYPE_INT64) {
        return RC_INVALID_VALUE;
    }

    if (input_shape->GetDimCount() != 1) {
        return RC_INVALID_VALUE;
    }

    uint32_t output_dim_count = input_shape->GetDim(0);
    std::vector<int64_t> output_dims(output_dim_count);
    for (size_t i = 0; i < output_dim_count; i++) {
        int64_t dim = input_host[i];
        if (dim < 0) {
            return RC_INVALID_VALUE;
        } else if (dim == 0) {
            output_shape->ReshapeAsScalar();
        }
        output_dims[i] = dim;
    }

    auto param = (const ConstantOfShapeParam*)arg;
    if (param->data_type == DATATYPE_UNKNOWN) {
        return RC_UNSUPPORTED;
    }

    output_shape->SetDataType(param->data_type);
    output_shape->Reshape(output_dims);
    return RC_SUCCESS;
}

RetCode ReshapeConstantOfShape(InputOutputInfo* info, const void* arg) {
    if (info->GetInputCount() != 1 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    auto input = info->GetInput<TensorImpl>(0);
    if (!input->GetBufferPtr()) {
        return RC_NOT_FOUND;
    }

    auto input_host = input->GetBufferPtr<int64_t>();
    return ReshapeConstantOfShape(info, arg, input_host);
}

}}} // namespace ppl::nn::oputils
