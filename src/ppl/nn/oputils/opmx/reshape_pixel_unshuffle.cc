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

#include "reshape_pixel_unshuffle.h"

#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace opmx {

ppl::common::RetCode ReshapePixelUnshuffle(InputOutputInfo* info, const ir::Attr* arg) {
    auto param = static_cast<const PixelUnshuffleParam*>(arg);

    const TensorShape& in_shape  = *info->GetInput<TensorImpl>(0)->GetShape();
    const int32_t out_dim_count = in_shape.GetDimCount();

    if (in_shape.GetDim(in_shape.GetDimCount() - 3) % param->scale_factor) {
        LOG(ERROR) << info->GetNode()->GetName() << " W dim[" << in_shape.GetDim(in_shape.GetDimCount() - 3)
                   << "] is not divisible by scale factor[" << param->scale_factor << "]";
        return RC_INVALID_VALUE;
    }

    if (in_shape.GetDim(in_shape.GetDimCount() - 2) % param->scale_factor) {
        LOG(ERROR) << info->GetNode()->GetName() << " H dim[" << in_shape.GetDim(in_shape.GetDimCount() - 2)
                   << "] is not divisible by scale factor[" << param->scale_factor << "]";
        return RC_INVALID_VALUE;
    }

    std::vector<int64_t> out_dims(out_dim_count);
    for (int32_t i = 0; i < out_dim_count - 3; i++) {
        out_dims[i] = in_shape.GetDim(i);
    }
    out_dims[out_dim_count - 3] = in_shape.GetDim(out_dim_count - 3) / param->scale_factor;
    out_dims[out_dim_count - 2] = in_shape.GetDim(out_dim_count - 2) / param->scale_factor;
    out_dims[out_dim_count - 1] = in_shape.GetDim(out_dim_count - 1) * param->scale_factor * param->scale_factor;

    info->GetOutput<TensorImpl>(0)->GetShape()->Reshape(out_dims);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::opmx
