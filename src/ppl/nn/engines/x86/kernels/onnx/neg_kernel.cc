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

#include "ppl/nn/engines/x86/kernels/onnx/neg_kernel.h"
#include "ppl/kernel/x86/fp32/neg.h"
#include "ppl/kernel/x86/int64/neg.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode NegKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(X, 0);
    PPLNN_X86_REQUIRED_OUTPUT(Y, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());

    PPLNN_X86_DEBUG_TRACE("Input [X]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(X);

    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    PPLNN_X86_REALLOC_TENSOR_BUFFER(Y);
    PPLNN_X86_DEBUG_TRACE("Output [Y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);

    const auto data_type = X->GetShape()->GetDataType();

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        return ppl::kernel::x86::neg_fp32(
            GetISA(), X->GetShape(), X->GetBufferPtr<float>(), Y->GetBufferPtr<float>());
    } if (data_type == ppl::common::DATATYPE_INT64) {
        return ppl::kernel::x86::neg_int64(
            GetISA(), X->GetShape(), X->GetBufferPtr<int64_t>(), Y->GetBufferPtr<int64_t>());
    } else {
        LOG(ERROR) << "unsupported datatype: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
