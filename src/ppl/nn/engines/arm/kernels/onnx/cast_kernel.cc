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

#include "ppl/nn/engines/arm/kernels/onnx/cast_kernel.h"
#include "ppl/kernel/arm_server/cast/neon/cast.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode CastKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [input]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_ARM_DEBUG_TRACE("Output [output]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_ARM_DEBUG_TRACE("to: %d\n", param_->to);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto input_data_type = input->GetShape()->GetDataType();
    const auto output_data_type = output->GetShape()->GetDataType();
    if ((input_data_type == ppl::common::DATATYPE_FLOAT16 || output_data_type == ppl::common::DATATYPE_FLOAT16) &&
        !MayUseISA(ppl::common::ISA_ARMV8_2)) {
        LOG(ERROR) << "fp16 needs isa >= armv8.2.";
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::kernel::arm_server::neon::cast(input->GetShape(), output->GetShape(), input->GetBufferPtr<void>(),
                                               output->GetBufferPtr<void>());
}

}}} // namespace ppl::nn::arm
