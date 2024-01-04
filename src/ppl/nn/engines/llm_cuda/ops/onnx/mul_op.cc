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

#include "mul_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/onnx/mul_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_add.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace onnx {

RetCode MulOp::CommonInit() {
    infer_type_and_format_func_ = GenericInferTypeAndFormat;
    infer_dims_func_ =  [](InputOutputInfo* info) -> RetCode {
        return ppl::nn::onnx::ReshapeAdd(info, nullptr);
    };
    return RC_SUCCESS;
}

RetCode MulOp::DoInit(const OptKernelOptions& options) {
    return CommonInit();
}

KernelImpl* MulOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<MulKernel>();
}

}}}}} // namespace ppl::nn::llm::cuda::pmx