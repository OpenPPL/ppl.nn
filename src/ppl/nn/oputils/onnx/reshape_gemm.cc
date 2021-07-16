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

#include "ppl/nn/oputils/onnx/reshape_gemm.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeGemm(InputOutputInfo* info, const void* arg) {
    if (info->GetInputCount() < 2) {
        return RC_INVALID_VALUE;
    }

    auto param = (const GemmParam*)arg;
    auto A = &info->GetInput<TensorImpl>(0)->GetShape();
    auto B = &info->GetInput<TensorImpl>(1)->GetShape();
    auto Y = &info->GetOutput<TensorImpl>(0)->GetShape();

    int32_t AMdim = 0;
    int32_t BNdim = 1;
    if (param->transA) {
        AMdim = 1;
    }
    if (param->transB) {
        BNdim = 0;
    }

    Y->Reshape({A->GetDim(AMdim), param->N == 0 ? B->GetDim(BNdim) : param->N});
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
