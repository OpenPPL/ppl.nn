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

#include "ppl/nn/oputils/onnx/reshape_maxunpool.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeMaxUnpool(InputOutputInfo* info, const void* arg) {
    auto param = (const MaxUnpoolParam*)arg;

    if (info->GetInputCount() != 2 && info->GetInputCount() != 3) {
        return RC_INVALID_VALUE;
    }
    if (info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    auto input_data = &info->GetInput<TensorImpl>(0)->GetShape();
    auto output = &info->GetOutput<TensorImpl>(0)->GetShape();

    const uint32_t dim_count = input_data->GetDimCount();
    std::vector<int64_t> output_dim(dim_count);
    for (uint32_t it = 0; it < dim_count; ++it) {
        output_dim[it] = input_data->GetDim(it);
    }

    std::vector<uint32_t> pads(dim_count * 2, 0);
    for (uint32_t i = 0; i < param->pads.size(); ++i) {
        pads[i] = param->pads[i];
    }

    // only hw or dhw
    for (int it = dim_count - 1; it > 1; --it) {
        output_dim[it] = (output_dim[it] - 1) * param->strides[it - 2] - pads[it - 2] - pads[it - 2 + dim_count - 2] +
            param->kernel_shape[it - 2];
    }
    output->Reshape(output_dim.data(), dim_count);
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
