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

#include "ppl/nn/engines/arm/kernels/onnx/pad_kernel.h"
#include "ppl/kernel/arm_server/pad/neon/pad.h"
#include "ppl/kernel/arm_server/common/memory.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode PadKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_ARM_REQUIRED_INPUT(x, 0);
    PPLNN_ARM_REQUIRED_INPUT(pads, 1);
    PPLNN_ARM_OPTIONAL_INPUT(constant, 2);
    PPLNN_ARM_REQUIRED_OUTPUT(y, 0);

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [x]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_ARM_DEBUG_TRACE("Output [y]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(y);
    PPLNN_ARM_DEBUG_TRACE("pad mode: %d\n", param_->mode);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    const int dim_count = x->GetShape()->GetDimCount();
    auto pads_data = pads->GetBufferPtr<int64_t>();
    auto start_pads = pads_data;
    auto end_pads = pads_data + dim_count;

    if (x->GetShape()->GetElementsExcludingPadding() ==
        y->GetShape()->GetElementsExcludingPadding()) { // no padding at all, just copy
        if (x->GetEdge()->CalcConsumerCount() == 1 && x->GetType() == TENSORTYPE_NORMAL) {
            y->TransferBufferFrom(x);
        } else {
            ppl::kernel::arm_server::memory_copy(x->GetBufferPtr(), x->GetShape()->GetBytesIncludingPadding(),
                                                 y->GetBufferPtr());
        }
        return ppl::common::RC_SUCCESS;
    }

    const void* constant_value = constant == nullptr ? nullptr : constant->GetBufferPtr<void>();

    switch (param_->mode) {
        case ppl::nn::onnx::PadParam::PAD_MODE_CONSTANT:
            return ppl::kernel::arm_server::neon::pad_constant(x->GetShape(), y->GetShape(), x->GetBufferPtr<void>(),
                                                               start_pads, end_pads, constant_value,
                                                               y->GetBufferPtr<void>());
        case ppl::nn::onnx::PadParam::PAD_MODE_REFLECT:
            return ppl::kernel::arm_server::neon::pad_reflect(x->GetShape(), y->GetShape(), x->GetBufferPtr<void>(),
                                                              start_pads, end_pads, constant_value,
                                                              y->GetBufferPtr<void>());
        case ppl::nn::onnx::PadParam::PAD_MODE_EDGE:
            return ppl::kernel::arm_server::neon::pad_edge(x->GetShape(), y->GetShape(), x->GetBufferPtr<void>(),
                                                           start_pads, end_pads, constant_value,
                                                           y->GetBufferPtr<void>());
        default:
            break;
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::arm
