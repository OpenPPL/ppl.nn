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

#include "reshape_moe_reduce.h"

#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace pmx {

ppl::common::RetCode ReshapeMoeReduce(InputOutputInfo* info, const ir::Attr* arg) {
    auto param = static_cast<const MoeReduceParam*>(arg);

    const TensorShape& y_expand_permute_shape = *info->GetInput<TensorImpl>(0)->GetShape();

    const uint32_t out_dim_count = y_expand_permute_shape.GetDimCount() - 1;

    if (y_expand_permute_shape.GetDim(y_expand_permute_shape.GetDimCount() - 2) != param->num_experts_per_token) {
        LOG(ERROR) << info->GetNode()->GetName() << " num_experts(" << param->num_experts_per_token
                   << ") not equal to scores's last dim("
                   << y_expand_permute_shape.GetDim(y_expand_permute_shape.GetDimCount() - 1) << ")";
        return RC_INVALID_VALUE;
    }

    std::vector<int64_t> out_dims(out_dim_count);
    for (uint32_t i = 0; i < out_dim_count - 1; ++i) {
        out_dims[i] = y_expand_permute_shape.GetDim(i);
    }
    out_dims[out_dim_count - 1] = y_expand_permute_shape.GetDim(y_expand_permute_shape.GetDimCount() - 1);

    info->GetOutput<TensorImpl>(0)->GetShape()->Reshape(out_dims);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::pmx
