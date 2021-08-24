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

#include <float.h>

#include "ppl/nn/engines/x86/optimizer/ops/onnx/lstm_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/lstm_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_lstm.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode LSTMOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    if (param_->activations.size() || param_->activation_alpha.size() || param_->activation_beta.size()) {
        LOG(ERROR) << "LSTM dose not support customize activations and parameters";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (param_->clip != FLT_MAX) {
        LOG(ERROR) << "LSTM dose not support clip";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (param_->input_forget) {
        LOG(ERROR) << "LSTM dose not support input_forget";
        return ppl::common::RC_UNSUPPORTED;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeLSTM(info, param_.get());
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

KernelImpl* LSTMOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<LSTMKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
