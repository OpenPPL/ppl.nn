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

#include "ppl/nn/oputils/pmx/reshape_rotary_position_embedding.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace pmx {

RetCode ReshapeRotaryPositionEmbedding(InputOutputInfo* info, const ir::Attr* arg) {
    auto& query_shape = *info->GetInput<TensorImpl>(0)->GetShape();
    auto& key_shape = *info->GetInput<TensorImpl>(1)->GetShape();

    info->GetOutput<TensorImpl>(0)->GetShape()->Reshape(query_shape.GetDims(), query_shape.GetDimCount());
    info->GetOutput<TensorImpl>(1)->GetShape()->Reshape(key_shape.GetDims(), key_shape.GetDimCount());

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::pmx
