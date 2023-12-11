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

#include "ppl/nn/oputils/pmx/reshape_dynamic_batching_key_value_cache.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace pmx {

RetCode ReshapeDynamicBatchingKeyValueCache(InputOutputInfo* info, const ir::Attr* arg, const int64_t kvlen) {
    const TensorShape& current_key_shape = *info->GetInput<TensorImpl>(0)->GetShape();
    const uint32_t out_dim_count = current_key_shape.GetDimCount();

    std::vector<int64_t> out_dims(out_dim_count);
    for (uint32_t i = 0; i < out_dim_count; ++i) {
        out_dims[i] = current_key_shape.GetDim(i);
    }

    out_dims[0] = kvlen;

    info->GetOutput<TensorImpl>(0)->GetShape()->Reshape(out_dims);
    info->GetOutput<TensorImpl>(1)->GetShape()->Reshape(out_dims);

    return RC_SUCCESS;
}

RetCode ReshapeDynamicBatchingKeyValueCache(InputOutputInfo* info, const ir::Attr* arg) {
    auto kvstarts_data = info->GetInput<TensorImpl>(3)->GetBufferPtr<int64_t>();
    if (!kvstarts_data) {
        LOG(DEBUG) << "ERROR: kvstarts's data is empty.";
        return RC_NOT_FOUND;
    }

    const TensorShape& kvstarts_shape = *info->GetInput<TensorImpl>(3)->GetShape();

    return ReshapeDynamicBatchingKeyValueCache(info, arg, kvstarts_data[kvstarts_shape.GetDim(0) - 1]);
}

}}} // namespace ppl::nn::pmx
