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

#include "ppl/nn/oputils/pmx/reshape_vision_embedding.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace pmx {

RetCode ReshapeVisionEmbedding(InputOutputInfo* info, const ir::Attr* arg) {
    const TensorShape& input = *info->GetInput<TensorImpl>(0)->GetShape();
    TensorShape& output = *info->GetOutput<TensorImpl>(0)->GetShape();
    auto param = static_cast<const VisionEmbeddingParam*>(arg);

    int64_t last_axis = input.GetDimCount() - 1;

    if (input.GetDim(last_axis) & 1) {
        LOG(DEBUG) << "last_dim(" << input.GetDim(last_axis) << ") of input must be an even number";
        return RC_INVALID_VALUE;
    }

    const uint32_t output_dim_count = 3;
    int64_t output_dims[output_dim_count];
    output_dims[0] = input.GetDims()[0];
    output_dims[1] = (param->image_size / param->patch_size) * (param->image_size / param->patch_size) + 1;
    output_dims[2] = param->hidden_dim;

    output.Reshape(output_dims, output_dim_count);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::pmx
