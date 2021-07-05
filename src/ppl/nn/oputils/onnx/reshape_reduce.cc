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

#include "ppl/nn/oputils/onnx/reshape_reduce.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeReduce(InputOutputInfo* info, const void* arg) {
    auto param = (const ReduceParam*)arg;
    auto x = &info->GetInput<TensorImpl>(0)->GetShape();
    auto y = &info->GetOutput<TensorImpl>(0)->GetShape();

    // check & prepare axes
    if (param->axes.size() > x->GetDimCount()) {
        return RC_INVALID_VALUE;
    }

    const uint32_t dim_count = x->GetDimCount();
    auto fixed_axes = param->axes;
    if (param->axes.empty()) { // empty axes means reduce all dimss
        y->ReshapeAsScalar();
        return RC_SUCCESS;
    }

    for (uint32_t i = 0; i < fixed_axes.size(); i++) {
        if (fixed_axes[i] >= (int)dim_count || fixed_axes[i] < -(int)dim_count) {
            return RC_INVALID_VALUE;
        }
        if (fixed_axes[i] < 0) { // turn negative axes to positive axes
            fixed_axes[i] = fixed_axes[i] + dim_count;
        }
    }

    // reshape
    y->Reshape(x->GetDims(), x->GetDimCount());
    if (param->keep_dims) {
        for (uint32_t a = 0; a < fixed_axes.size(); ++a) {
            y->SetDim(fixed_axes[a], 1);
        }
    } else {
        for (uint32_t a = 0; a < fixed_axes.size(); ++a) {
            y->SetDim(fixed_axes[a] - a, 0);
            for (size_t i = fixed_axes[a] + 1; i < x->GetDimCount(); ++i) {
                y->SetDim(i - a - 1, x->GetDim(i));
            }
            y->SetDimCount(y->GetDimCount() - 1);
        }
        if (y->GetDimCount() == 0) {
            y->ReshapeAsScalar();
        }
    }
    y->CalcPadding();

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
