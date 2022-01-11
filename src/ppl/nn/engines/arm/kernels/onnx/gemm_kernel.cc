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
    PPLNN_ARM_DEBUG_TRACE("trans_A: %d\n", param_->transA);
    PPLNN_ARM_DEBUG_TRACE("trans_B: %d\n", param_->transB);
    PPLNN_ARM_DEBUG_TRACE("alpha: %f\n", param_->alpha);
    PPLNN_ARM_DEBUG_TRACE("beta: %f\n", param_->beta);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = A->GetShape()->GetDataType();
    if ((data_type != ppl::common::DATATYPE_FLOAT32 &&
         data_type != ppl::common::DATATYPE_FLOAT16    )                  ||
        A->GetShape()->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY     ) {
        LOG(ERROR) << "only support fp32/fp16 ndarray now.";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (param_->transA || param_->transB) {
        LOG(ERROR) << "only support anbn now.";
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
    const int32_t N = param_->N == 0 ? B->GetShape()->GetDim(BNdim) : param_->N;

    void *src_A = A->GetBufferPtr<void>();
    void *src_B = B->GetBufferPtr<void>();
    void *dst_Y = Y->GetBufferPtr<void>();
    int32_t lda = param_->transA ? M : K;
    int32_t ldb = param_->transB ? K : N;
    int32_t ldy = N;
    auto alpha = param_->alpha;
    auto beta = param_->beta;
    auto trans_A = param_->transA;
    auto trans_B = param_->transB;
    auto isa_flag = GetISA();

    uint32_t fuse_type;
    if (gemm_fuse_relu_) {
        fuse_type = ppl::kernel::arm_server::neon::gemm_fuse_flag::RELU;
    } else {
        fuse_type = ppl::kernel::arm_server::neon::gemm_fuse_flag::NONE;
    }
    void *src_C = nullptr;
    auto c_type = ppl::kernel::arm_server::neon::gemm_C_type::EMPTY;
    // TODO: fully support gemm op
    (void)trans_A;
    (void)trans_B;
    (void)isa_flag;
    (void)fuse_type;
    (void)c_type;

    auto ldc = 0;
    if (C != nullptr && !C->GetShape()->IsEmpty()) {
        src_C = C->GetBufferPtr<float>();
        if (C->GetShape()->GetDimCount() == 2) {
            if (C->GetShape()->GetDim(0) == Y->GetShape()->GetDim(0) && C->GetShape()->GetDim(1) == Y->GetShape()->GetDim(1)) {
                c_type = ppl::kernel::arm_server::neon::gemm_C_type::MATRIX;
                ldc = C->GetShape()->GetDim(1);
            }
        }
    }

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        if (MayUseISA(ppl::common::ISA_ARMV8)) {
            sgemm_m1 = 80;
            sgemm_n1 = 32;
            sgemm_k1 = 128;
            sgemm_m3 = 2560;
            sgemm_k3 = 5120;

            BufferDesc tmp_buffer_desc;

            uint64_t tmp_buffer_size = ppl::kernel::arm_server::neon::ppl_arm_server_kernel_fp32_gemm_get_buffer_size(
                sgemm_m1, sgemm_n1);

            auto status = GetArmDevice()->AllocTmpBuffer(tmp_buffer_size, &tmp_buffer_desc);
            if (status != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_size << "] for kernel[" << GetName()
                        << "] failed: " << ppl::common::GetRetCodeStr(status);
                return status;
            }
            return ppl::kernel::arm_server::neon::gemm_fp32(
                (const float *)src_A,
                (const float *)src_B,
                (const float *)src_C,
                (float *)dst_Y,
                (float *)tmp_buffer_desc.addr,
                M,
                N,
                K,
                lda,
                ldb,
                ldc,
                ldy,
                alpha,
                beta,
                sgemm_m1,
                sgemm_n1,
                sgemm_k1,
                sgemm_m3,
                sgemm_k3);
        }
        else {
            LOG(ERROR) << "unsupported datatype: " << ppl::common::GetDataTypeStr(data_type) 
                    << "with isa " << GetISA() << ".";
            return ppl::common::RC_UNSUPPORTED;
        }
    }
#ifdef PPL_USE_ARM_SERVER_FP16
    else if (data_type == ppl::common::DATATYPE_FLOAT16) {
        if (MayUseISA(ppl::common::ISA_ARMV8_2)) {
            sgemm_m1 = 80;
            sgemm_n1 = 64;
            sgemm_k1 = 128;
            sgemm_m3 = 2560;
            sgemm_k3 = 5120;

            BufferDesc tmp_buffer_desc;

            uint64_t tmp_buffer_size = ppl::kernel::arm_server::neon::ppl_arm_server_kernel_fp16_gemm_get_buffer_size(
                sgemm_m1, sgemm_n1);

            auto status = GetArmDevice()->AllocTmpBuffer(tmp_buffer_size, &tmp_buffer_desc);
            if (status != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_size << "] for kernel[" << GetName()
                        << "] failed: " << ppl::common::GetRetCodeStr(status);
                return status;
            }
            return ppl::kernel::arm_server::neon::gemm_fp16(
                (const __fp16 *)src_A,
                (const __fp16 *)src_B,
                (const __fp16 *)src_C,
                (__fp16 *)dst_Y,
                (__fp16 *)tmp_buffer_desc.addr,
                M,
                N,
                K,
                lda,
                ldb,
                ldc,
                ldy,
                alpha,
                beta,
                sgemm_m1,
                sgemm_n1,
                sgemm_k1,
                sgemm_m3,
                sgemm_k3);
        }
        else {
            LOG(ERROR) << "unsupported datatype: " << ppl::common::GetDataTypeStr(data_type) 
                    << "with isa " << GetISA() << ".";
            return ppl::common::RC_UNSUPPORTED;
        }
    }
#endif
    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::arm
