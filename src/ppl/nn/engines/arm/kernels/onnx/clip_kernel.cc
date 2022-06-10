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

#include <stdint.h>
#include <float.h>

#include "ppl/nn/engines/arm/kernels/onnx/clip_kernel.h"
#include "ppl/kernel/arm_server/clip/neon/clip.h"

namespace ppl { namespace nn { namespace arm {

bool ClipKernel::CanDoExecute(const KernelExecContext& ctx) const {
    auto tensor = ctx.GetInput<TensorImpl>(0);
    if (!tensor || tensor->GetShape()->GetBytesIncludingPadding() == 0) {
        return false;
    }
    return true;
}

ppl::common::RetCode ClipKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_ARM_REQUIRED_INPUT(input, 0);
    PPLNN_ARM_OPTIONAL_INPUT(min_tensor, 1);
    PPLNN_ARM_OPTIONAL_INPUT(max_tensor, 2);
    PPLNN_ARM_REQUIRED_OUTPUT(output, 0);

    float pmin_val = -FLT_MAX;
    float pmax_val = FLT_MAX;

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [input]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(input);
    if (min_tensor) {
        pmin_val = (min_tensor->GetBufferPtr<float>())[0];
        PPLNN_ARM_DEBUG_TRACE("Input [min]:\n");
        PPL_ARM_TENSOR_PRINT_DEBUG_MSG(min_tensor);
    }
    if (max_tensor) {
        pmax_val = (max_tensor->GetBufferPtr<float>())[0];
        PPLNN_ARM_DEBUG_TRACE("Input [max]:\n");
        PPL_ARM_TENSOR_PRINT_DEBUG_MSG(max_tensor);
    }
    PPLNN_ARM_DEBUG_TRACE("Output [output]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = input->GetShape()->GetDataType();
    if (data_type == ppl::common::DATATYPE_FLOAT16 && !MayUseISA(ppl::common::ISA_ARMV8_2)) {
        LOG(ERROR) << "fp16 needs isa >= armv8.2.";
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::kernel::arm_server::neon::clip(
        input->GetShape(), input->GetBufferPtr<void>(), min_tensor ? min_tensor->GetBufferPtr<void>() : nullptr,
        max_tensor ? max_tensor->GetBufferPtr<void>() : nullptr, pmin_val, pmax_val, output->GetBufferPtr<void>());
}

}}} // namespace ppl::nn::arm
