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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/batch_normalization_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/batch_normalization_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_batch_normalization.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

BatchNormalizationOp::BatchNormalizationOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeBatchNormalization(info, param_.get());
    };

    infer_type_func_ = GenericInferType;
}

RetCode BatchNormalizationOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode BatchNormalizationOp::SelectFormat(const InputOutputInfo& info,
                                           std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                           std::vector<ppl::common::dataformat_t>* selected_output_formats) {
    selected_input_formats->at(0) = selected_output_formats->at(0) =
        info.GetInput<TensorImpl>(0)->GetShape()->GetDataFormat();
    for (uint32_t i = 1; i < info.GetInputCount(); i++) {
        selected_input_formats->at(i) = ppl::common::DATAFORMAT_NDARRAY;
    }
    return RC_SUCCESS;
}

KernelImpl* BatchNormalizationOp::CreateKernelImpl() const {
    auto kernel = CreateKernelImplWithParam<BatchNormalizationKernel>(param_.get());
    if (kernel) {
        kernel->SetFuseReLU(fuse_relu_);
    }
    return kernel;
}

}}} // namespace ppl::nn::arm
