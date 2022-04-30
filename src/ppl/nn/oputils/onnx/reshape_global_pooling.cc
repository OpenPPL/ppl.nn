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

#include "ppl/nn/oputils/onnx/reshape_global_pooling.h"
#include "ppl/common/log.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include <cmath>

using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ReshapeGlobalPooling(InputOutputInfo* info) {
    auto x = info->GetInput<TensorImpl>(0)->GetShape();
    auto y = info->GetOutput<TensorImpl>(0)->GetShape();

    y->SetDimCount(x->GetDimCount());
    y->SetDim(0, x->GetDim(0));
    y->SetDim(1, x->GetDim(1));
    const int32_t kernel_dims = x->GetDimCount() - 2;
    for (int32_t i = 2; i < kernel_dims + 2; ++i) {
        y->SetDim(i, std::min((int64_t)1l, x->GetDim(i))); // input tensor dim may be zero
    }
    y->CalcPadding();
    if (info->GetOutputCount() == 2) {
        auto z = info->GetOutput<TensorImpl>(1)->GetShape();
        z->SetDataType(DATATYPE_INT64);
        z->Reshape(y->GetDims(), y->GetDimCount());
    }
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
