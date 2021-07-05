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

#include "ppl/nn/oputils/onnx/reshape_non_zero.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeNonZero(InputOutputInfo* info, const void*) {
    if (info->GetInputCount() != 1 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    auto x = &info->GetInput<TensorImpl>(0)->GetShape();
    auto y = &info->GetOutput<TensorImpl>(0)->GetShape();

    const uint32_t input_dim_count = x->GetDimCount();
    const uint32_t max_output_num = x->GetElementsExcludingPadding();
    y->Reshape({input_dim_count, max_output_num});

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
