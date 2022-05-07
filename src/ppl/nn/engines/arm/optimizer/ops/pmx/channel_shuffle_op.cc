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

#include "ppl/nn/engines/arm/optimizer/ops/pmx/channel_shuffle_op.h"
#include "ppl/nn/engines/arm/kernels/pmx/channel_shuffle_kernel.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

ChannelShuffleOp::ChannelShuffleOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_type_func_ = GenericInferType;

    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        auto& input0 = *info->GetInput<TensorImpl>(0)->GetShape();
        int64_t channels = input0.GetDim(1);
        for (uint32_t i = 1; i < info->GetInputCount(); ++i) {
            channels += info->GetInput<TensorImpl>(1)->GetShape()->GetDim(1);
        }
        if (channels % info->GetOutputCount()) {
            return ppl::common::RC_INVALID_VALUE;
        }
        channels /= info->GetOutputCount();
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto& output = *info->GetOutput<TensorImpl>(i)->GetShape();
            output.Reshape(input0.GetDims(), input0.GetRealDimCount());
            output.SetDim(1, channels);
        }

        return RC_SUCCESS;
    };
}

RetCode ChannelShuffleOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode ChannelShuffleOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                                       vector<dataformat_t>* selected_output_formats) {
    auto data_format = info.GetInput<TensorImpl>(0)->GetShape()->GetDataFormat();
    if (info.GetInputCount() == 2 && info.GetInput<TensorImpl>(1)->GetShape()->GetDataFormat() != data_format) {
        data_format = DATAFORMAT_NDARRAY;
    }
    for (uint32_t i = 0; i < info.GetInputCount(); i++) {
        selected_input_formats->at(i) = data_format;
    }
    for (uint32_t i = 0; i < info.GetOutputCount(); i++) {
        selected_output_formats->at(i) = data_format;
    }

    return RC_SUCCESS;
}

KernelImpl* ChannelShuffleOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ChannelShuffleKernel>(param_.get());
}

}}} // namespace ppl::nn::arm
