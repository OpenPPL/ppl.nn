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

#include "ppl/nn/oputils/mmcv/reshape_mmcv_gridsample.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeMMCVGridSample(InputOutputInfo* info, const void* arg) {
    if (info->GetInputCount() != 2 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    auto input0 = &info->GetInput<TensorImpl>(0)->GetShape();
    auto input1 = &info->GetInput<TensorImpl>(1)->GetShape();
    auto output = &info->GetOutput<TensorImpl>(0)->GetShape();

    if (input0->GetDimCount() != 4) {
        return RC_INVALID_VALUE;
    }

    if (input1->GetDimCount() != 4 || input1->GetDim(3) != 2) {
        return RC_INVALID_VALUE;
    }

    const int32_t out_n = input0->GetDim(0);
    const int32_t out_c = input0->GetDim(1);
    const int32_t out_h = input1->GetDim(1);
    const int32_t out_w = input1->GetDim(2);

    output->Reshape({out_n, out_c, out_h, out_w});
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
