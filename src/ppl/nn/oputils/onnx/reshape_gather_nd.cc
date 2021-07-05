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

#include "ppl/nn/oputils/onnx/reshape_gather_nd.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeGatherND(InputOutputInfo* info, const void*) {
    if (info->GetInputCount() != 2 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    auto input_data = &info->GetInput<TensorImpl>(0)->GetShape();
    auto input_indices = &info->GetInput<TensorImpl>(1)->GetShape();
    auto output = &info->GetOutput<TensorImpl>(0)->GetShape();

    const uint32_t r = input_data->GetRealDimCount();
    const uint32_t q = input_indices->GetRealDimCount();
    if (r < 1 || q < 1) {
        return RC_INVALID_VALUE;
    }
    const uint32_t last_indices_dim = input_indices->GetDim(q - 1);
    if (last_indices_dim < 1 || last_indices_dim > r) {
        return RC_INVALID_VALUE;
    }
    const uint32_t output_dim_count = q + r - last_indices_dim - 1;
    output->SetDimCount(output_dim_count);
    size_t i = 0;
    for (i = 0; i < q - 1; i++) {
        output->SetDim(i, input_indices->GetDim(i));
    }
    for (; i < output_dim_count; i++) {
        output->SetDim(i, input_data->GetDim(i - (q - 1) + last_indices_dim));
    }
    output->CalcPadding();
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
