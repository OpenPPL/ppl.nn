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

#include "ppl/nn/oputils/onnx/reshape_range.h"
#include <algorithm>
#include <cmath>
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeRange(InputOutputInfo* info, double start, double limit, double delta) {
    auto output = info->GetOutput<TensorImpl>(0);
    const uint32_t num_elements = std::max((int)(std::ceil((limit - start) / delta)), 0);
    output->GetShape().Reshape({num_elements});
    return RC_SUCCESS;
}

common::RetCode ReshapeRange(InputOutputInfo* info, const void*) {
    if (info->GetInputCount() != 3 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    if (!info->GetInput<TensorImpl>(0)->GetBufferPtr() || !info->GetInput<TensorImpl>(1)->GetBufferPtr() ||
        !info->GetInput<TensorImpl>(2)->GetBufferPtr()) {
        return RC_NOT_FOUND;
    }

    const auto data_type = info->GetInput<TensorImpl>(0)->GetShape().GetDataType();
    double start, limit, delta;
    if (data_type == DATATYPE_INT64) {
        start = *info->GetInput<TensorImpl>(0)->GetBufferPtr<int64_t>();
        limit = *info->GetInput<TensorImpl>(1)->GetBufferPtr<int64_t>();
        delta = *info->GetInput<TensorImpl>(2)->GetBufferPtr<int64_t>();
    } else if (data_type == DATATYPE_FLOAT32) {
        start = *info->GetInput<TensorImpl>(0)->GetBufferPtr<float>();
        limit = *info->GetInput<TensorImpl>(1)->GetBufferPtr<float>();
        delta = *info->GetInput<TensorImpl>(2)->GetBufferPtr<float>();
    } else {
        return RC_UNSUPPORTED;
    }

    auto output = info->GetOutput<TensorImpl>(0);
    const uint32_t num_elements = std::max(std::ceil((limit - start) / delta), 0.0);
    output->GetShape().Reshape({num_elements});

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
