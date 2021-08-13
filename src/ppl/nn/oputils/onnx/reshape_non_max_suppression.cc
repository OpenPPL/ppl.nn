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

#include "ppl/nn/oputils/onnx/reshape_non_max_suppression.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeNonMaxSuppression(InputOutputInfo* info, int64_t max_output_boxes_per_class) {
    if (info->GetInputCount() < 2 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    const TensorShape& input_boxes = info->GetInput<TensorImpl>(0)->GetShape();
    const TensorShape& input_scores = info->GetInput<TensorImpl>(1)->GetShape();
    auto output = &info->GetOutput<TensorImpl>(0)->GetShape();

    if (input_boxes.GetDimCount() != 3 || input_scores.GetDimCount() != 3) {
        return RC_INVALID_VALUE;
    }

    if (input_boxes.GetDim(2) != 4 || input_boxes.GetDim(0) != input_scores.GetDim(0) ||
        input_boxes.GetDim(1) != input_scores.GetDim(2)) {
        return RC_INVALID_VALUE;
    }

    const int64_t batch = input_scores.GetDim(0);
    const int64_t num_classes = input_scores.GetDim(1);
    const int64_t num_boxes = input_scores.GetDim(2);
    const int64_t num_max_output = std::min(max_output_boxes_per_class, num_boxes) * batch * num_classes;

    output->Reshape({num_max_output, 3});
    return RC_SUCCESS;
}

RetCode ReshapeNonMaxSuppression(InputOutputInfo* info) {
    auto max_output_boxes_per_class = info->GetInputCount() > 2 ? info->GetInput<TensorImpl>(2) : nullptr;
    return ReshapeNonMaxSuppression(info, max_output_boxes_per_class ? max_output_boxes_per_class->GetBufferPtr<int64_t>()[0] : 0);
}

}}} // namespace ppl::nn::oputils
