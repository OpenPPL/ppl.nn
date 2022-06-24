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

#include "ppl/nn/engines/x86/kernels/onnx/pow_kernel.h"
#include "ppl/nn/common/logger.h"

#include "ppl/kernel/x86/fp32/arithmetic_max6d.h"
#include "ppl/kernel/x86/common/memory.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode PowKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(A, 0);
    PPLNN_X86_REQUIRED_INPUT(B, 1);
    PPLNN_X86_REQUIRED_OUTPUT(C, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());

    PPLNN_X86_DEBUG_TRACE("Input [A]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(A);
    PPLNN_X86_DEBUG_TRACE("Input [B]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(B);

    PPLNN_X86_REALLOC_TENSOR_BUFFER(C);
    PPLNN_X86_DEBUG_TRACE("Output [C]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(C);

    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = A->GetShape()->GetDataType();

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        if (B->GetShape()->IsScalar() && B->GetBufferPtr<float>()[0] == 1.0f) {
            return ppl::kernel::x86::memory_copy(
                A->GetBufferPtr<float>(),
                A->GetShape()->CalcBytesIncludingPadding(),
                C->GetBufferPtr<float>());
        }

        if (B->GetShape()->IsScalar() && B->GetBufferPtr<float>()[0] == 2.0f) {
            if (MayUseISA(ppl::common::ISA_X86_AVX)) {
                return ppl::kernel::x86::mul_ndarray_max6d_fp32_avx(
                    A->GetShape(),
                    A->GetShape(),
                    A->GetBufferPtr<float>(),
                    A->GetBufferPtr<float>(),
                    C->GetBufferPtr<float>());
            } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
                return ppl::kernel::x86::mul_ndarray_max6d_fp32_sse(
                    A->GetShape(),
                    A->GetShape(),
                    A->GetBufferPtr<float>(),
                    A->GetBufferPtr<float>(),
                    C->GetBufferPtr<float>());
            }
        }

        if (MayUseISA(ppl::common::ISA_X86_AVX)) {
            return ppl::kernel::x86::pow_ndarray_max6d_fp32_avx(
                A->GetShape(),
                B->GetShape(),
                A->GetBufferPtr<float>(),
                B->GetBufferPtr<float>(),
                C->GetBufferPtr<float>());
        } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
            return ppl::kernel::x86::pow_ndarray_max6d_fp32_sse(
                A->GetShape(),
                B->GetShape(),
                A->GetBufferPtr<float>(),
                B->GetBufferPtr<float>(),
                C->GetBufferPtr<float>());
        } else {
            LOG(ERROR) << "get unsupported isa " << GetISA();
        }
    } else {
        LOG(ERROR) << "unsupported datatype: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
