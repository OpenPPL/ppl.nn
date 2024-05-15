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

#include "ppl/nn/oputils/opmx/reshape_moe_row_parallel_linear.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace opmx {

RetCode ReshapeMoeRowParallelLinear(InputOutputInfo* info, const ir::Attr* arg, int64_t world_size,
                                       int64_t in_features_pack_size, int64_t out_features_pack_size,
                                       bool check_weight_shape) {
    auto param = static_cast<const MoeRowParallelLinearParam*>(arg);
    const TensorShape& input_shape = *info->GetInput<TensorImpl>(0)->GetShape();
    const TensorShape& offset_shape = *info->GetInput<TensorImpl>(1)->GetShape();
    const TensorShape& weight_shape = *info->GetInput<TensorImpl>(2)->GetShape();
    const uint32_t out_dim_count = input_shape.GetDimCount();
    const uint32_t weight_dim_count = weight_shape.GetDimCount();

    if (param->in_features % world_size) {
        LOG(ERROR) << "in_features % world_size != 0, " << param->in_features << " % " << world_size;
        return RC_INVALID_VALUE;
    }

    int64_t in_features_per_part = param->in_features / world_size;

    if (param->input_is_parallel) {
        if (input_shape.GetDim(out_dim_count - 1) != in_features_per_part) {
            LOG(ERROR) << info->GetNode()->GetName() <<
                " in_features_per_part(" << in_features_per_part <<
                ") not equal to input's last dim(" <<
                input_shape.GetDim(out_dim_count - 1) << "), " <<
                "world_size = " << world_size;
            return RC_INVALID_VALUE;
        }
    } else {
        if (input_shape.GetDim(out_dim_count - 1) != param->in_features) {
            LOG(ERROR) << info->GetNode()->GetName() <<
                " in_features(" << param->in_features <<
                ") not equal to input's last dim(" <<
                input_shape.GetDim(out_dim_count - 1) << ")";
            return RC_INVALID_VALUE;
        }
    }

    if(check_weight_shape && weight_shape.GetDim(weight_dim_count - 1) * in_features_pack_size != in_features_per_part) {
        LOG(ERROR) << info->GetNode()->GetName() <<
                " in_features_per_part(" << in_features_per_part <<
                ") not equal to weight's in feature dim(" <<
                weight_shape.GetDim(weight_dim_count - 1) * in_features_pack_size << "), " <<
                "world_size = " << world_size;
        return RC_INVALID_VALUE;
    }

    if (check_weight_shape && offset_shape.GetDim(0) != weight_shape.GetDim(0) + 1) {
        LOG(ERROR) << info->GetNode()->GetName() << " export_offset_shape(" << offset_shape.GetDim(0)
                   << ") not equal to weight's first dim + 1(" << weight_shape.GetDim(0) + 1 << ")";
        return RC_INVALID_VALUE;
    }

    std::vector<int64_t> out_dims(out_dim_count);
    for (uint32_t i = 0; i < out_dim_count; ++i) {
        out_dims[i] = input_shape.GetDim(i);
    }
    out_dims[out_dim_count - 1] = param->out_features;
    info->GetOutput<TensorImpl>(0)->GetShape()->Reshape(out_dims);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::opmx
