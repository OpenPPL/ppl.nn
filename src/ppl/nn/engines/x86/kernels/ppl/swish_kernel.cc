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

#include "ppl/nn/engines/x86/kernels/ppl/swish_kernel.h"
#include "ppl/nn/common/logger.h"

#include "ppl/kernel/x86/fp32/swish.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode SwishKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [input]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_X86_DEBUG_TRACE("beta: %f\n", param_->beta);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const ppl::common::datatype_t data_type = input->GetShape().GetDataType();

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        if (MayUseISA(ppl::common::ISA_X86_FMA)) {
            return ppl::kernel::x86::swish_fp32_fma(&input->GetShape(), input->GetBufferPtr<float>(), param_->beta,
                                                    output->GetBufferPtr<float>());
        } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
            return ppl::kernel::x86::swish_fp32_sse(&input->GetShape(), input->GetBufferPtr<float>(), param_->beta,
                                                    output->GetBufferPtr<float>());
        } else {
            return ppl::kernel::x86::swish_fp32(&input->GetShape(), input->GetBufferPtr<float>(), param_->beta,
                                                output->GetBufferPtr<float>());
        }
    } else {
        LOG(ERROR) << "unsupported data type " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
