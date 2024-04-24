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

#include "ppl/nn/oputils/opmx/reshape_parallel_embedding.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace opmx {

RetCode ReshapeParallelEmbedding(InputOutputInfo* info, const ir::Attr* arg, int64_t world_size) {
    auto param = static_cast<const ParallelEmbeddingParam*>(arg);
    const TensorShape& input_shape = *info->GetInput<TensorImpl>(0)->GetShape();
    const TensorShape& weight_shape = *info->GetInput<TensorImpl>(1)->GetShape();
    const uint32_t out_dim_count = input_shape.GetDimCount() + 1;

    if (param->embedding_dims % world_size) {
        LOG(ERROR) << "embedding_dims % world_size != 0, " << param->embedding_dims << " % " << world_size;
        return RC_INVALID_VALUE;
    }

    auto embedding_dims_per_part = param->embedding_dims / world_size;
    if(weight_shape.GetDim(1) != embedding_dims_per_part) {
        LOG(ERROR) << info->GetNode()->GetName() <<
                " embedding_dims_per_part(" << embedding_dims_per_part <<
                ") not equal to weight's embedding dim(" <<
                weight_shape.GetDim(1) << "), " <<
                "world_size = " << world_size;
        return RC_INVALID_VALUE;
    }

    std::vector<int64_t> out_dims(out_dim_count);
    for (uint32_t i = 0; i < out_dim_count - 1; ++i) {
        out_dims[i] = input_shape.GetDim(i);
    }
    out_dims[out_dim_count - 1] = param->embedding_dims;

    info->GetOutput<TensorImpl>(0)->GetShape()->Reshape(out_dims);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::opmx
