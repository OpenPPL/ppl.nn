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

#include "ppl/nn/oputils/onnx/reshape_topk.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeTopK(InputOutputInfo* info, const void* arg, int64_t k) {
    auto param = (const TopKParam*)arg;

    auto x = info->GetInput<TensorImpl>(0);
    auto values = info->GetOutput<TensorImpl>(0);
    auto indices = info->GetOutput<TensorImpl>(1);

    const int32_t dim_count = x->GetShape().GetDimCount();
    int32_t axis = param->axis;

    if (axis < -dim_count || axis > dim_count - 1) {
        return RC_INVALID_VALUE;
    }
    if (axis < 0) {
        axis += dim_count;
    }

    const int64_t axis_dim = x->GetShape().GetDim(axis);
    // if (k > axis_dim) {
    //     return RC_INVALID_VALUE;
    // }

    values->GetShape().Reshape(x->GetShape().GetDims(), x->GetShape().GetDimCount());
    indices->GetShape().Reshape(x->GetShape().GetDims(), x->GetShape().GetDimCount());
    values->GetShape().SetDim(axis, std::min(k, axis_dim));
    indices->GetShape().SetDim(axis, std::min(k, axis_dim));
    values->GetShape().CalcPadding();
    indices->GetShape().CalcPadding();

    return RC_SUCCESS;
}

RetCode ReshapeTopK(InputOutputInfo* info, const void* arg) {
    if (info->GetInputCount() != 2 || info->GetOutputCount() != 2) {
        return RC_INVALID_VALUE;
    }

    auto k_ptr = info->GetInput<TensorImpl>(1)->GetBufferPtr<int64_t>();
    if (!k_ptr) {
        return RC_NOT_FOUND;
    }

    return ReshapeTopK(info, arg, *k_ptr);
}

}}} // namespace ppl::nn::oputils
