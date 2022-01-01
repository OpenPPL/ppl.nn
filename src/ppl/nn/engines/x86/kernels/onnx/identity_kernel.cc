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

#include "ppl/nn/engines/x86/kernels/onnx/identity_kernel.h"
#include "ppl/kernel/x86/common/memory.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode IdentityKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(input, 0);
    PPLNN_X86_REQUIRED_OUTPUT(output, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());

    PPLNN_X86_DEBUG_TRACE("Input [input]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(input);

    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    if (input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL) {
        output->TransferBufferFrom(input);
        PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
    } else {
        PPLNN_X86_REALLOC_TENSOR_BUFFER(output);
        PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
        return ppl::kernel::x86::memory_copy(input->GetBufferPtr(), input->GetShape()->GetBytesIncludingPadding(), output->GetBufferPtr());
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::x86
