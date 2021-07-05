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

#include "ppl/nn/oputils/onnx/reshape_ceil.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeCeil(InputOutputInfo* info, const void* arg) {
    if (info->GetInputCount() != 1 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    auto input = &info->GetInput<TensorImpl>(0)->GetShape();
    auto output = &info->GetOutput<TensorImpl>(0)->GetShape();

    output->SetDataType(DATATYPE_FLOAT32);
    if (input->IsScalar()) {
        output->ReshapeAsScalar();
    } else {
        output->Reshape(input->GetDims(), input->GetDimCount());
    }
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
