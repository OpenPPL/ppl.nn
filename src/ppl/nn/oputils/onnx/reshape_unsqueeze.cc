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

#include "ppl/nn/oputils/onnx/reshape_unsqueeze.h"
#include <algorithm>
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeUnsqueeze(InputOutputInfo* info, const void* arg) {
    auto param = (const UnsqueezeParam*)arg;
    std::vector<int32_t> axes(param->axes.size());

    const TensorShape& input = info->GetInput<TensorImpl>(0)->GetShape();
    const int32_t out_dim_count = (int32_t)input.GetRealDimCount() + param->axes.size();

    for (uint32_t i = 0; i < param->axes.size(); ++i) {
        if (param->axes[i] < (int32_t)(-out_dim_count) || param->axes[i] >= (int32_t)out_dim_count) {
            return RC_INVALID_VALUE;
        }
        if (param->axes[i] < 0) {
            axes[i] = out_dim_count + param->axes[i];
        } else {
            axes[i] = param->axes[i];
        }
    }

    std::sort(axes.begin(), axes.end());
    std::vector<int64_t> output_dim(out_dim_count);
    for (int32_t oid = 0, aid = 0, iid = 0; oid < out_dim_count; ++oid) {
        if (aid < (int32_t)axes.size() && oid == axes[aid]) {
            output_dim[oid] = 1;
            ++aid;
        } else {
            output_dim[oid] = input.GetDim(iid);
            ++iid;
        }
    }

    info->GetOutput<TensorImpl>(0)->GetShape().Reshape(output_dim);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
