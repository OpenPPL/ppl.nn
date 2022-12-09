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

#include "ppl/nn/engines/x86/optimizer/ops/onnx/random_uniform_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/random_uniform_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_random_uniform.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode RandomUniformOp::DoInit(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    if (this->param_->dtype != DATATYPE_FLOAT32) {
        LOG(ERROR) << "only support fp32 now.";
        return RC_UNSUPPORTED;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeRandomUniform(info, param_.get());
    };

    infer_type_func_ = [this](InputOutputInfo* info) -> void {
        info->GetOutput<TensorImpl>(0)->GetShape()->SetDataType(this->param_->dtype);
    };

    return RC_SUCCESS;
}

KernelImpl* RandomUniformOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<RandomUniformKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
