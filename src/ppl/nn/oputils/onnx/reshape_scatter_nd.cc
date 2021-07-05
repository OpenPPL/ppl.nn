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

#include "ppl/nn/oputils/onnx/reshape_scatter_nd.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeScatterND(InputOutputInfo* info, const void*) {
    if (info->GetInputCount() != 3 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    const TensorShape& input_data = info->GetInput<TensorImpl>(0)->GetShape();
    const TensorShape& input_indices = info->GetInput<TensorImpl>(1)->GetShape();
    const TensorShape& input_updates = info->GetInput<TensorImpl>(2)->GetShape();
    auto output = &info->GetOutput<TensorImpl>(0)->GetShape();

    const uint32_t r = input_data.GetRealDimCount();
    const uint32_t q = input_indices.GetRealDimCount();
    if (r < 1 || q < 1) {
        return RC_INVALID_VALUE;
    }
    const uint32_t k = input_indices.GetDim(q - 1);
    if (k < 1 || k > r) {
        return RC_INVALID_VALUE;
    }
    const uint32_t updates_dim_count = q + r - k - 1;
    if (input_updates.GetDimCount() != updates_dim_count) {
        return RC_INVALID_VALUE;
    }
    for (uint32_t i = 0; i < q - 1; i++) {
        if (input_indices.GetDim(i) != input_updates.GetDim(i)) {
            return RC_INVALID_VALUE;
        }
    }
    for (uint32_t i = q - 1; i < input_updates.GetDimCount(); i++) {
        if (input_updates.GetDim(i) != input_data.GetDim(i - (q - 1) + k)) {
            return RC_INVALID_VALUE;
        }
    }
    output->Reshape(input_data.GetDims(), input_data.GetDimCount());
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
