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

#include "ppl/nn/engines/x86/kernels/ppl/channel_shuffle_kernel.h"
#include "ppl/kernel/x86/fp32/channel_shuffle.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode ChannelShuffleKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(X, 0);
    PPLNN_X86_OPTIONAL_INPUT(X1, 1);
    PPLNN_X86_REQUIRED_OUTPUT(Y, 0);
    PPLNN_X86_OPTIONAL_OUTPUT(Y1, 1);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [X]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(X);
    if (X1) {
        PPLNN_X86_DEBUG_TRACE("Input [X1]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(X1);
    }
    PPLNN_X86_DEBUG_TRACE("Output [Y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);
    if (Y1) {
        PPLNN_X86_DEBUG_TRACE("Output [Y1]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y1);
    }
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    if (!(ctx->GetInputCount() == ctx->GetOutputCount()) &&
        !(ctx->GetInputCount() == 2 && ctx->GetOutputCount() == 1)) {
        LOG(ERROR) << "output tensor count and input tensor count must be equal or only have one output tensor.";
        return ppl::common::RC_UNSUPPORTED;
    }

    int group_ = param_->group;

    if (X->GetShape().GetDimCount() != 4) {
        LOG(ERROR) << "incorrect input dimcount: " << X->GetShape().GetDimCount();
        return ppl::common::RC_UNSUPPORTED;
    }

    if (X->GetShape().GetDim(1) % group_) {
        LOG(ERROR) << "unsupported ChanneShuffle group: " << group_;
        return ppl::common::RC_UNSUPPORTED;
    }

    if (X->GetShape().GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        if (X->GetShape().GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
            if (X1) {
                return kernel::x86::channel_shuffle_ndarray_concat_split_fp32(
                    &X->GetShape(), &X1->GetShape(),
                    X->GetBufferPtr<float>(), X1->GetBufferPtr<float>(),
                    group_,
                    Y->GetBufferPtr<float>(), Y1 ? Y1->GetBufferPtr<float>() : nullptr);
            } else {
                return kernel::x86::channel_shuffle_ndarray_fp32(
                    &X->GetShape(), X->GetBufferPtr<float>(), group_,
                    Y->GetBufferPtr<float>());
            }
        } else if (X->GetShape().GetDataFormat() == ppl::common::DATAFORMAT_N16CX) {
            if (X1) {
                return kernel::x86::channel_shuffle_n16cx_concat_split_fp32(
                    &X->GetShape(), &X1->GetShape(),
                    X->GetBufferPtr<float>(), X1->GetBufferPtr<float>(),
                    group_,
                    Y->GetBufferPtr<float>(), Y1 ? Y1->GetBufferPtr<float>() : nullptr);
            } else {
                return kernel::x86::channel_shuffle_n16cx_fp32(
                    &X->GetShape(), X->GetBufferPtr<float>(), group_,
                    Y->GetBufferPtr<float>());
            }
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(X->GetShape().GetDataFormat());
        }
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(X->GetShape().GetDataType());
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
