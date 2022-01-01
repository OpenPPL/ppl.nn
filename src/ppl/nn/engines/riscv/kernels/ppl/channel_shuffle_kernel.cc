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
#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"

#include "ppl/nn/engines/riscv/kernels/ppl/channel_shuffle_kernel.h"
#include "ppl/kernel/riscv/fp32/channel_shuffle.h"
#include "ppl/kernel/riscv/fp16/channel_shuffle.h"

namespace ppl { namespace nn { namespace riscv {

ppl::common::RetCode ChannelShuffleKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_RISCV_REQUIRED_INPUT(X, 0);
    PPLNN_RISCV_OPTIONAL_INPUT(X1, 1);
    PPLNN_RISCV_REQUIRED_OUTPUT(Y, 0);
    PPLNN_RISCV_OPTIONAL_OUTPUT(Y1, 1);

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [X]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(X);
    if (X1) {
        PPLNN_RISCV_DEBUG_TRACE("Input [X1]:\n");
        PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(X1);
    }
    PPLNN_RISCV_DEBUG_TRACE("Output [Y]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(Y);
    if (Y1) {
        PPLNN_RISCV_DEBUG_TRACE("Output [Y1]:\n");
        PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(Y1);
    }

    if (!(ctx->GetInputCount() == ctx->GetOutputCount()) &&
        !(ctx->GetInputCount() == 2 && ctx->GetOutputCount() == 1)) {
        LOG(ERROR) << "output tensor count and input tensor count must be equal or only have one output tensor.";
        return ppl::common::RC_UNSUPPORTED;
    }

    int64_t group_ = param_->group;

    if (X->GetShape()->GetDimCount() != 4) {
        LOG(ERROR) << "incorrect input dimcount: " << X->GetShape()->GetDimCount();
        return ppl::common::RC_UNSUPPORTED;
    }

    if (X->GetShape()->GetDim(1) % group_) {
        LOG(ERROR) << "unsupported ChanneShuffle group: " << group_;
        return ppl::common::RC_UNSUPPORTED;
    }

    if (X->GetShape()->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        if (X->GetShape()->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
            if (X1) {
                LOG(DEBUG) << "Channels Shuffle Ndarray concat-split !";
                return kernel::riscv::channel_shuffle_ndarray_concat_split_fp32(
                    X->GetShape(), X1->GetShape(), X->GetBufferPtr<float>(), X1->GetBufferPtr<float>(), group_,
                    Y->GetBufferPtr<float>(), Y1 ? Y1->GetBufferPtr<float>() : nullptr);
            } else {
                LOG(DEBUG) << "Channel Shuffle Ndarray only !";
                return kernel::riscv::channel_shuffle_ndarray_fp32(X->GetShape(), X->GetBufferPtr<float>(), group_,
                                                                   Y->GetBufferPtr<float>());
            }
        } else if (X->GetShape()->GetDataFormat() == ppl::common::DATAFORMAT_N4CX) {
            if (X1) {
                LOG(DEBUG) << "Channel Shuffle Select concat-split !";
                return kernel::riscv::channel_shuffle_n4cx_concat_split_fp32(
                    X->GetShape(), X1->GetShape(), X->GetBufferPtr<float>(), X1->GetBufferPtr<float>(), group_,
                    Y->GetBufferPtr<float>(), Y1 ? Y1->GetBufferPtr<float>() : nullptr);
            } else {
                LOG(DEBUG) << "Channel Shuffle Select Here !";
                return kernel::riscv::channel_shuffle_n4cx_fp32(X->GetShape(), X->GetBufferPtr<float>(), group_,
                                                                Y->GetBufferPtr<float>());
            }
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(X->GetShape()->GetDataFormat());
        }
    } else if (X->GetShape()->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        if (X->GetShape()->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
            if (X1) {
                return kernel::riscv::channel_shuffle_ndarray_concat_split_fp16(
                    X->GetShape(), X1->GetShape(), X->GetBufferPtr<__fp16>(), X1->GetBufferPtr<__fp16>(), group_,
                    Y->GetBufferPtr<__fp16>(), Y1 ? Y1->GetBufferPtr<__fp16>() : nullptr);
            } else {
                return kernel::riscv::channel_shuffle_ndarray_fp16(X->GetShape(), X->GetBufferPtr<__fp16>(), group_,
                                                                   Y->GetBufferPtr<__fp16>());
            }
        } else if (X->GetShape()->GetDataFormat() == ppl::common::DATAFORMAT_N8CX) {
            if (X1) {
                return kernel::riscv::channel_shuffle_n8cx_concat_split_fp16(
                    X->GetShape(), X1->GetShape(), X->GetBufferPtr<__fp16>(), X1->GetBufferPtr<__fp16>(), group_,
                    Y->GetBufferPtr<__fp16>(), Y1 ? Y1->GetBufferPtr<__fp16>() : nullptr);
            } else {
                return kernel::riscv::channel_shuffle_n8cx_fp16(X->GetShape(), X->GetBufferPtr<__fp16>(), group_,
                                                                Y->GetBufferPtr<__fp16>());
            }
        }
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(X->GetShape()->GetDataType());
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::riscv
