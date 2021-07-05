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

#include "ppl/nn/oputils/onnx/reshape_flatten.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeFlatten(InputOutputInfo* info, const void* arg) {
    auto param = (const FlattenParam*)arg;
    if (info->GetInputCount() != 1 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    auto input = info->GetInput<TensorImpl>(0);
    auto output = info->GetOutput<TensorImpl>(0);

    const int32_t dim_count = input->GetShape().GetDimCount();
    if (param->axis < -dim_count || param->axis > dim_count) {
        return RC_INVALID_VALUE;
    }

    const int32_t axis = param->axis < 0 ? param->axis + dim_count : param->axis;

    int64_t outer_dim = 1;
    for (int32_t i = 0; i < axis; i++) {
        outer_dim *= input->GetShape().GetDim(i);
    }
    int64_t inner_dim = 1;
    for (int32_t i = axis; i < dim_count; i++) {
        inner_dim *= input->GetShape().GetDim(i);
    }

    output->GetShape().Reshape({outer_dim, inner_dim});
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
