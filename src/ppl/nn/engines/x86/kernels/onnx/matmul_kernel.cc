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

#include "ppl/nn/engines/x86/kernels/onnx/matmul_kernel.h"
#include "ppl/nn/utils/destructor.h"
#include "ppl/kernel/x86/fp32/matmul.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode MatMulKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(A, 0);
    PPLNN_X86_REQUIRED_INPUT(B, 1);
    PPLNN_X86_REQUIRED_OUTPUT(Y, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [A]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(A);
    PPLNN_X86_DEBUG_TRACE("Input [B]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(B);

    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    PPLNN_X86_REALLOC_TENSOR_BUFFER(Y);
    PPLNN_X86_DEBUG_TRACE("Output [Y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);

    const auto data_type = A->GetShape()->GetDataType();
    const auto data_format = A->GetShape()->GetDataFormat();

    if (data_type == ppl::common::DATATYPE_FLOAT32 && data_format == ppl::common::DATAFORMAT_NDARRAY) {
        return kernel::x86::matmul_ndarray_fp32(
            GetISA(), A->GetShape(), B->GetShape(), Y->GetShape(),
            A->GetBufferPtr<float>(), B->GetBufferPtr<float>(), Y->GetBufferPtr<float>());
    } else {
        LOG(ERROR) << "only support fp32 ndarray now.";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
