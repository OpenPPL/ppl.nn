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

#include "ppl/nn/oputils/onnx/reshape_ms_deformable_attention.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace pmx {

RetCode ReshapeMSDeformAttn(InputOutputInfo* info, const ir::Attr* arg) {
    auto param = static_cast<const MSDeformAttnParam*>(arg);

    auto input_data = info->GetInput<TensorImpl>(0)->GetShape();

    auto spatial_shapes = info->GetInput<TensorImpl>(1)->GetShape();
    auto sampling_loc = info->GetInput<TensorImpl>(3)->GetShape();
    
    auto batch = input_data->GetDim(0);

    const int im2col_step_ = batch<param->im2col_step? batch: param->im2col_step;

    if(batch % im2col_step_ != 0){
        LOG(DEBUG) << "batch = "<<batch<<" must divide im2_col_step = " << im2col_step_;
        return RC_INVALID_VALUE;
    }

    // auto spatial_size = input_data->GetDim(1);
    auto num_heads = input_data->GetDim(2);
    auto channels = input_data->GetDim(3);

    auto num_levels = spatial_shapes->GetDim(0);
    auto num_query = sampling_loc->GetDim(1);

    auto output = info->GetOutput<TensorImpl>(0)->GetShape();

    std::vector<int64_t> output_dim = {batch, num_query, num_heads*channels}; // add one dimension in axis

    info->GetOutput<TensorImpl>(0)->GetShape()->Reshape(output_dim);
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
