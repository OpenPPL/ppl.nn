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

#include "ppl/nn/engines/arm/kernels/onnx/not_kernel.h"
#include "ppl/kernel/arm_server/not/neon/not.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode NotKernel::DoExecute(KernelExecContext* ctx) {
    auto x = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [x]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_ARM_DEBUG_TRACE("Output [output]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = x->GetShape()->GetDataType();
    if (data_type == ppl::common::DATATYPE_FLOAT16 && !MayUseISA(ppl::common::ISA_ARMV8_2)) {
        LOG(ERROR) << "fp16 needs isa >= armv8.2.";
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::kernel::arm_server::neon::not_bool(x->GetShape(), x->GetBufferPtr<uint8_t>(), output->GetBufferPtr<uint8_t>());
}

}}} // namespace ppl::nn::arm
