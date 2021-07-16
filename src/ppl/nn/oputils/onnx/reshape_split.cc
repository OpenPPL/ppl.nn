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

#include "ppl/nn/oputils/onnx/reshape_split.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeSplit(InputOutputInfo* info, const void* arg) {
    auto param = (const SplitParam*)arg;
    const TensorShape& shape = info->GetInput<TensorImpl>(0)->GetShape();
    int dim_count = shape.GetDimCount();
    uint32_t split_axis = param->axis >= 0 ? param->axis : param->axis + dim_count;
    if (param->split_point.size() == 0) {
        uint32_t in_dim = shape.GetDim(split_axis);
        uint32_t split_dim = in_dim / info->GetOutputCount();
        if (split_dim * info->GetOutputCount() != in_dim) {
            return RC_INVALID_VALUE;
        }

        std::vector<int64_t> tmp_output_dim(dim_count);
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            for (int it = 0; it < dim_count; ++it) {
                tmp_output_dim[it] = shape.GetDim(it);
            }
            tmp_output_dim[split_axis] = split_dim;
            info->GetOutput<TensorImpl>(i)->GetShape().Reshape(tmp_output_dim);
        }
        return RC_SUCCESS;
    }
    if (info->GetOutputCount() != param->split_point.size()) {
        return RC_INVALID_VALUE;
    }

    uint32_t sumDim = 0;
    for (uint32_t i = 0; i < param->split_point.size(); ++i) {
        sumDim += param->split_point[i];
    }
    if (sumDim != shape.GetDim(split_axis)) {
        return RC_INVALID_VALUE;
    }

    std::vector<int64_t> tmp_output_dim(dim_count);
    for (uint32_t i = 0; i < param->split_point.size(); ++i) {
        for (int it = 0; it < dim_count; ++it) {
            tmp_output_dim[it] = shape.GetDim(it);
        }
        tmp_output_dim[split_axis] = param->split_point[i];
        info->GetOutput<TensorImpl>(i)->GetShape().Reshape(tmp_output_dim);
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
