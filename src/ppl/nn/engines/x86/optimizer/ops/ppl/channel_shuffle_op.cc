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

#include "ppl/nn/engines/x86/optimizer/ops/ppl/channel_shuffle_op.h"
#include "ppl/nn/engines/x86/kernels/ppl/channel_shuffle_kernel.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode ChannelShuffleOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    
    infer_type_func_ = GenericInferType;
    infer_dims_func_ = GenericInferDims;
    return RC_SUCCESS;
}

void ChannelShuffleOp::SetGroup(int group) {
    param_->group = group;
}

RetCode ChannelShuffleOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                                       vector<dataformat_t>* selected_output_formats) {
    auto input_format = info.GetInput<TensorImpl>(0)->GetShape().GetDataFormat();

    if (input_format == DATAFORMAT_N16CX) {
        selected_input_formats->at(0) = DATAFORMAT_N16CX;
        selected_output_formats->at(0) = DATAFORMAT_N16CX;
    }

    if (input_format == DATAFORMAT_NDARRAY) {
        selected_input_formats->at(0) = DATAFORMAT_NDARRAY;
        selected_output_formats->at(0) = DATAFORMAT_NDARRAY;
    }

    return RC_SUCCESS;
}

KernelImpl* ChannelShuffleOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ChannelShuffleKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
