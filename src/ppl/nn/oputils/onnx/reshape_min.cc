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

#include "ppl/nn/oputils/onnx/reshape_min.h"
#include "ppl/nn/oputils/broadcast.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeMin(InputOutputInfo* info, const void*) {
    if (info->GetInputCount() < 2) {
        return RC_INVALID_VALUE;
    }

    auto out = &info->GetOutput<TensorImpl>(0)->GetShape();

    MultiInputBroadCaster multi_input_bc;
    for (uint32_t i = 0; i < info->GetInputCount(); i++) {
        multi_input_bc.PushBackInputTensorShape(info->GetInput<TensorImpl>(i)->GetShape());
    }
    multi_input_bc.CalcBroadCast();
    if (!multi_input_bc.CanBroadCast()) {
        return RC_INVALID_VALUE;
    }

    auto& output_shape = multi_input_bc.OutputTensorShape();
    if (output_shape.IsScalar()) {
        out->ReshapeAsScalar();
    } else {
        out->Reshape(output_shape.GetDims(), output_shape.GetDimCount());
    }
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
