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

#include "ppl/nn/engines/arm/kernels/pmx/channel_shuffle_kernel.h"
#include "ppl/kernel/arm_server/channel_shuffle/neon/channel_shuffle.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode ChannelShuffleKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_ARM_REQUIRED_INPUT(X, 0);
    PPLNN_ARM_OPTIONAL_INPUT(X1, 1);
    PPLNN_ARM_REQUIRED_OUTPUT(Y, 0);
    PPLNN_ARM_OPTIONAL_OUTPUT(Y1, 1);

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [X]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(X);
    if (X1) {
        PPLNN_ARM_DEBUG_TRACE("Input [X1]:\n");
        PPL_ARM_TENSOR_PRINT_DEBUG_MSG(X1);
    }
    PPLNN_ARM_DEBUG_TRACE("Output [Y]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(Y);
    if (Y1) {
        PPLNN_ARM_DEBUG_TRACE("Output [Y1]:\n");
        PPL_ARM_TENSOR_PRINT_DEBUG_MSG(Y1);
    }
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    if (!X1) {
        LOG(ERROR) << "arm channel shuffle only support fuse_concat now.";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (Y1) {
        return ppl::kernel::arm_server::neon::channel_shuffle_concat_split(
            X->GetShape(), X1->GetShape(), Y->GetShape(), Y1->GetShape(), X->GetBufferPtr<void>(),
            X1->GetBufferPtr<void>(), param_->group, Y->GetBufferPtr<void>(), Y1->GetBufferPtr<void>());
    } else {
        return ppl::kernel::arm_server::neon::channel_shuffle_concat(X->GetShape(), X1->GetShape(), Y->GetShape(),
                                                                     X->GetBufferPtr<void>(), X1->GetBufferPtr<void>(),
                                                                     param_->group, Y->GetBufferPtr<void>());
    }
}

}}} // namespace ppl::nn::arm
