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

#include "ppl/nn/engines/x86/kernels/onnx/gemm_kernel.h"
#include "ppl/nn/utils/destructor.h"
#include "ppl/kernel/x86/fp32/gemm.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode GemmKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(A, 0);
    PPLNN_X86_REQUIRED_INPUT(B, 1);
    PPLNN_X86_OPTIONAL_INPUT(C, 2);
    PPLNN_X86_REQUIRED_OUTPUT(Y, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [A]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(A);
    PPLNN_X86_DEBUG_TRACE("Input [B]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(B);
    if (C) {
        PPLNN_X86_DEBUG_TRACE("Input [C]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(C);
    }

    auto M = A->GetShape()->GetDim(0 + param_->trans_a);
    auto K = A->GetShape()->GetDim(1 - param_->trans_a);
    auto N = B->GetShape()->GetDim(1 - param_->trans_b);
    auto isa = GetISA();

    PPLNN_X86_DEBUG_TRACE("trans_A: %d\n", param_->trans_a);
    PPLNN_X86_DEBUG_TRACE("trans_B: %d\n", param_->trans_b);
    PPLNN_X86_DEBUG_TRACE("alpha: %f\n", param_->alpha);
    PPLNN_X86_DEBUG_TRACE("beta: %f\n", param_->beta);
    PPLNN_X86_DEBUG_TRACE("post: %d\n", param_->post);
    PPLNN_X86_DEBUG_TRACE("packed_b: %p\n", param_->packed_b);
    PPLNN_X86_DEBUG_TRACE("M, N, K: %ld, %ld, %ld\n", M ,N, K);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", isa);

    PPLNN_X86_REALLOC_TENSOR_BUFFER(Y);
    PPLNN_X86_DEBUG_TRACE("Output [Y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);

    if (A->GetShape()->GetDataType() != ppl::common::DATATYPE_FLOAT32 ||
        A->GetShape()->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY) {
        LOG(ERROR) << "only support fp32 ndarray now.";
        return ppl::common::RC_UNSUPPORTED;
    }

    auto A_data = A->GetBufferPtr<const float>();
    auto B_data = param_->packed_b ? param_->packed_b : B->GetBufferPtr<const float>();
    auto Y_data = Y->GetBufferPtr<float>();

    auto typeA = param_->trans_a ? ppl::kernel::x86::gemm_m_type::TRANS : ppl::kernel::x86::gemm_m_type::NOTRANS;
    auto typeB = param_->trans_b ? ppl::kernel::x86::gemm_m_type::TRANS : ppl::kernel::x86::gemm_m_type::NOTRANS;
    typeB = param_->packed_b ? ppl::kernel::x86::gemm_m_type::PACKED : typeB;

    auto lda = A->GetShape()->GetDim(1);
    auto ldb = B->GetShape()->GetDim(1);
    auto ldy = Y->GetShape()->GetDim(1);
    auto ldsum = ldy;

    const float *C_data = nullptr;
    const float *sum_data = nullptr;
    const float *bias_data = nullptr;
    ppl::kernel::x86::gemm_m_type_t typesum = ppl::kernel::x86::gemm_m_type::EMPTY;
    ppl::kernel::x86::gemm_v_type_t typebias = ppl::kernel::x86::gemm_v_type::EMPTY;
    if (C) {
        C_data = C->GetBufferPtr<const float>();
        if (C->GetShape()->CalcElementsExcludingPadding() == 1) {
            typebias = ppl::kernel::x86::gemm_v_type::SCALAR;
        } else if (C->GetShape()->GetDimCount() == 1) {
            typebias = ppl::kernel::x86::gemm_v_type::ROW_VEC;
        } else if (C->GetShape()->GetDimCount() == 2) {
            if (C->GetShape()->GetDim(0) == 1) {
                typebias = ppl::kernel::x86::gemm_v_type::ROW_VEC;
            } else if (C->GetShape()->GetDim(1) == 1) {
                typebias = ppl::kernel::x86::gemm_v_type::COL_VEC;
            } else {
                typesum = ppl::kernel::x86::gemm_m_type::NOTRANS;
            }
        }
        if (typesum) {
            sum_data = C_data;
            ldsum = C->GetShape()->GetDim(1);
        }
        if (typebias) {
            bias_data = C_data;
        }
    }

    return ppl::kernel::x86::gemm_fp32(
        isa, A_data, B_data, bias_data, sum_data,
        typeA, typeB, typebias, typesum, M, N, K,
        lda, ldb, ldy, ldsum, param_->alpha, 0.0f,
        param_->beta, param_->beta, param_->post, Y_data);
}

}}} // namespace ppl::nn::x86
