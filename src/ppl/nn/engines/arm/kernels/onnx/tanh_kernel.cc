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

#include "ppl/nn/engines/arm/kernels/onnx/tanh_kernel.h"
#include "ppl/kernel/arm_server/tanh/neon/tanh.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode TanHKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [input]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_ARM_DEBUG_TRACE("Output [output]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = input->GetShape()->GetDataType();

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        if (MayUseISA(ppl::common::ISA_ARMV8)) {
            return ppl::kernel::arm_server::neon::tanh_fp32(input->GetShape(), input->GetBufferPtr<float>(),
                                                            output->GetBufferPtr<float>());
        }
    }
#ifdef PPLNN_USE_ARMV8_2_FP16
    else if (data_type == ppl::common::DATATYPE_FLOAT16) {
        if (MayUseISA(ppl::common::ISA_ARMV8_2)) {
            return ppl::kernel::arm_server::neon::tanh_fp16(input->GetShape(), input->GetBufferPtr<__fp16>(),
                                                            output->GetBufferPtr<__fp16>());
        }
    }
#endif
    else {
        LOG(ERROR) << "unsupported datatype: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::arm
