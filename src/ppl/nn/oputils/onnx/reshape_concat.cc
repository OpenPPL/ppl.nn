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

#include "ppl/nn/oputils/onnx/reshape_concat.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeConcat(InputOutputInfo* info, const void* arg) {
    auto param = (const ConcatParam*)arg;
    const TensorShape& in_shape0 = info->GetInput<TensorImpl>(0)->GetShape();
    uint32_t fixed_axis =
        param->axis >= 0 ? param->axis : param->axis + info->GetInput<TensorImpl>(0)->GetShape().GetDimCount();

    std::vector<int64_t> output_dim(in_shape0.GetDimCount());
    for (uint32_t i = 0; i < in_shape0.GetDimCount(); ++i) {
        if (i == fixed_axis) {
            output_dim[i] = 0;
            for (uint32_t j = 0; j < info->GetInputCount(); ++j) {
                output_dim[i] += info->GetInput<TensorImpl>(j)->GetShape().GetDim(i);
            }
        } else {
            for (uint32_t j = 1; j < info->GetInputCount(); ++j) {
                if (info->GetInput<TensorImpl>(j)->GetShape().GetDim(i) != in_shape0.GetDim(i)) {
                    return RC_INVALID_VALUE;
                }
            }
            output_dim[i] = in_shape0.GetDim(i);
        }
    }

    info->GetOutput<TensorImpl>(0)->GetShape().Reshape(output_dim);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
