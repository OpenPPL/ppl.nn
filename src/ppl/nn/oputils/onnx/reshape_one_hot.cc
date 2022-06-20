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

#include "ppl/nn/oputils/onnx/reshape_one_hot.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ReshapeOneHot(InputOutputInfo* info, const ir::Attr* arg) {
    auto param = static_cast<const OneHotParam*>(arg);
    const TensorShape& in_shape0 = *info->GetInput<TensorImpl>(0)->GetShape(); // indices
    const auto* depth_ptr = info->GetInput<TensorImpl>(1)->GetBufferPtr<const int64_t>(); // depth
    if (!depth_ptr)
        return RC_OTHER_ERROR; // no data is ok during graph preprocess but not inference.
    const int64_t depth = depth_ptr[0];
    uint32_t real_axis = // [-r-1, r]
        param->axis >= 0 ? param->axis : param->axis + in_shape0.GetDimCount() + 1;
    std::vector<int64_t> output_dim; // add one dimension in axis
    for (uint32_t i = 0; i < real_axis; ++i) {
        output_dim.push_back(in_shape0.GetDim(i));
    }
    output_dim.push_back(depth);
    for (uint32_t i = real_axis; i < in_shape0.GetDimCount(); ++i) {
        output_dim.push_back(in_shape0.GetDim(i));
    }
    info->GetOutput<TensorImpl>(0)->GetShape()->Reshape(output_dim);
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
