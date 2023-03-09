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
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"
#include <algorithm>
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ReshapeUnsqueeze(InputOutputInfo* info, const ir::Attr* arg, const int64_t* axes) {
    auto axes_size = info->GetInput<TensorImpl>(1)->GetShape()->GetDim(0);
    std::vector<int32_t> vector_axes(axes_size);


    const TensorShape& input = *info->GetInput<TensorImpl>(0)->GetShape();
    const int32_t out_dim_count = (int32_t)input.GetRealDimCount() + axes_size;

    for (uint32_t i = 0; i < axes_size; ++i) {
        if (axes[i] < (int32_t)(-out_dim_count) || axes[i] >= (int32_t)out_dim_count) {
            LOG(DEBUG) << "ERROR: axes[" << i << "]'s value[" << axes[i] << "] is out of range["
                       << -out_dim_count << ", " << out_dim_count << "].";
            return RC_INVALID_VALUE;
        }
        if (axes[i] < 0) {
            vector_axes[i] = out_dim_count + axes[i];
        } else {
            vector_axes[i] = axes[i];
        }
    }

    std::sort(vector_axes.begin(), vector_axes.end());
    std::vector<int64_t> output_dim(out_dim_count);
    for (int32_t oid = 0, aid = 0, iid = 0; oid < out_dim_count; ++oid) {
        if (aid < (int32_t)vector_axes.size() && oid == vector_axes[aid]) {
            output_dim[oid] = 1;
            ++aid;
        } else {
            output_dim[oid] = input.GetDim(iid);
            ++iid;
        }
    }

    info->GetOutput<TensorImpl>(0)->GetShape()->Reshape(output_dim);

    return RC_SUCCESS;
}

RetCode ReshapeUnsqueeze(InputOutputInfo* info, const ir::Attr* arg) {
    if (info->GetInputCount() != 2) {
        LOG(DEBUG) << "ERROR: input count[" << info->GetInputCount() << "] != 2.";
        return RC_INVALID_VALUE;
    }

    auto axes = info->GetInput<TensorImpl>(1)->GetBufferPtr<int64_t>();
    return ReshapeUnsqueeze(info, arg, axes);
}

}}} // namespace ppl::nn::onnx
