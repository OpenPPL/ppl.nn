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

#include "ppl/nn/engines/arm/kernels/onnx/gemm_kernel.h"
#include "ppl/kernel/arm_server/gemm/neon/gemm.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode GemmKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_ARM_REQUIRED_INPUT(A, 0);
    PPLNN_ARM_REQUIRED_INPUT(B, 1);
    PPLNN_ARM_OPTIONAL_INPUT(C, 2);
    PPLNN_ARM_REQUIRED_OUTPUT(Y, 0);

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [A]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(A);
    PPLNN_ARM_DEBUG_TRACE("Input [B]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(B);
    if (C) {
        PPLNN_ARM_DEBUG_TRACE("Input [C]:\n");
        PPL_ARM_TENSOR_PRINT_DEBUG_MSG(C);
    }
    PPLNN_ARM_DEBUG_TRACE("Output [Y]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(Y);
    PPLNN_ARM_DEBUG_TRACE("transA: %d\n", param_->transA);
    PPLNN_ARM_DEBUG_TRACE("transB: %d\n", param_->transB);
    PPLNN_ARM_DEBUG_TRACE("alpha: %f\n", param_->alpha);
    PPLNN_ARM_DEBUG_TRACE("beta: %f\n", param_->beta);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = A->GetShape()->GetDataType();
    if ((data_type != ppl::common::DATATYPE_FLOAT32 && data_type != ppl::common::DATATYPE_FLOAT16) ||
        A->GetShape()->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY) {
        LOG(ERROR) << "only support fp32/fp16 ndarray now.";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (data_type == ppl::common::DATATYPE_FLOAT16 && !MayUseISA(ppl::common::ISA_ARMV8_2)) {
        LOG(ERROR) << "fp16 needs isa >= armv8.2.";
        return ppl::common::RC_UNSUPPORTED;
    }

    int32_t AMdim = 0;
    int32_t AKdim = 1;
    if (param_->transA) {
        AMdim = 1;
        AKdim = 0;
    }
    int32_t BNdim = 1;
    if (param_->transB) {
        BNdim = 0;
    }

    const int32_t M = A->GetShape()->GetDim(AMdim);
    const int32_t K = A->GetShape()->GetDim(AKdim);
    const int32_t N = B->GetShape()->GetDim(BNdim);

    void* src_A = A->GetBufferPtr<void>();
    void* src_B = B->GetBufferPtr<void>();
    void* dst_Y = Y->GetBufferPtr<void>();
    int32_t lda = param_->transA ? M : K;
    int32_t ldb = param_->transB ? K : N;
    int32_t ldy = N;
    auto alpha = param_->alpha;
    auto beta = param_->beta;
    auto transA = param_->transA;
    auto transB = param_->transB;

    void* src_C = nullptr;
    auto c_type = ppl::kernel::arm_server::neon::gemm_C_type::EMPTY;

    auto ldc = 0;
    if (C != nullptr && !C->GetShape()->IsEmpty()) {
        src_C = C->GetBufferPtr<void>();
        if (C->GetShape()->GetElementsExcludingPadding() == 1) {
            c_type = ppl::kernel::arm_server::neon::gemm_C_type::SCALAR;
        } else if (C->GetShape()->GetDimCount() == 1) {
            c_type = ppl::kernel::arm_server::neon::gemm_C_type::VECTOR_W;
            ldc = C->GetShape()->GetDim(0);
        } else if (C->GetShape()->GetDimCount() == 2) {
            if (C->GetShape()->GetDim(0) == 1) {
                c_type = ppl::kernel::arm_server::neon::gemm_C_type::VECTOR_W;
            } else if (C->GetShape()->GetDim(1) == 1) {
                c_type = ppl::kernel::arm_server::neon::gemm_C_type::VECTOR_H;
            } else {
                c_type = ppl::kernel::arm_server::neon::gemm_C_type::MATRIX;
            }
            ldc = C->GetShape()->GetDim(1);
        }
    }

    return ppl::kernel::arm_server::neon::gemm_ndarray(src_A, src_B, src_C, data_type, M, N, K, lda, ldb, ldc, transA,
                                                       transB, alpha, beta, ldy, c_type, dst_Y);
}

}}} // namespace ppl::nn::arm
