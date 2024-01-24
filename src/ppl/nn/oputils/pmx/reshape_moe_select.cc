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

#include "reshape_moe_select.h"

#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace pmx {

ppl::common::RetCode ReshapeMoeSelect(InputOutputInfo* info, const ir::Attr* arg) {
    auto param = static_cast<const MoeSelectParam*>(arg);

    const TensorShape& x_shape = *info->GetInput<TensorImpl>(0)->GetShape();
    const TensorShape& scores_shape = *info->GetInput<TensorImpl>(1)->GetShape();

    const uint32_t out_dim_count0 = x_shape.GetDimCount() + 1;
    const uint32_t out_dim_count1 = x_shape.GetDimCount();

    if (scores_shape.GetDim(scores_shape.GetDimCount() - 1) != param->num_experts) {
        LOG(ERROR) << info->GetNode()->GetName() << " num_experts[" << param->num_experts
                   << "] not equal to scores's last dim[" << scores_shape.GetDim(scores_shape.GetDimCount() - 1) << "]";
        return RC_INVALID_VALUE;
    }

    std::vector<int64_t> x_expand_permute_dims(out_dim_count0);
    for (uint32_t i = 0; i < out_dim_count0 - 2; ++i) {
        x_expand_permute_dims[i] = x_shape.GetDim(i);
    }
    x_expand_permute_dims[out_dim_count0 - 2] = param->num_experts_per_token;
    x_expand_permute_dims[out_dim_count0 - 1] = x_shape.GetDim(x_shape.GetDimCount() - 1);

    std::vector<int64_t> expert_weight_dims(out_dim_count1);
    for (uint32_t i = 0; i < out_dim_count1 - 1; ++i) {
        expert_weight_dims[i] = x_shape.GetDim(i);
    }
    expert_weight_dims[out_dim_count1 - 1] = param->num_experts_per_token;

    std::vector<int64_t> expert_offset_dims(1);
    expert_offset_dims[0] = param->num_experts + 1;

    info->GetOutput<TensorImpl>(0)->GetShape()->Reshape(x_expand_permute_dims);
    info->GetOutput<TensorImpl>(1)->GetShape()->Reshape(expert_weight_dims);
    info->GetOutput<TensorImpl>(2)->GetShape()->Reshape(expert_weight_dims);
    info->GetOutput<TensorImpl>(3)->GetShape()->Reshape(expert_offset_dims);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::pmx
