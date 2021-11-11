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
#include "ppl/kernel/x86/fp32/gemm_v2.h"

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
    PPLNN_X86_DEBUG_TRACE("Output [Y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);
    PPLNN_X86_DEBUG_TRACE("trans_A: %d\n", param_->transA);
    PPLNN_X86_DEBUG_TRACE("trans_B: %d\n", param_->transB);
    PPLNN_X86_DEBUG_TRACE("alpha: %f\n", param_->alpha);
    PPLNN_X86_DEBUG_TRACE("beta: %f\n", param_->beta);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    if (A->GetShape().GetDataType() != ppl::common::DATATYPE_FLOAT32 ||
        A->GetShape().GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY) {
        LOG(ERROR) << "only support fp32 ndarray now.";
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

    const int32_t M = A->GetShape().GetDim(AMdim);
    const int32_t K = A->GetShape().GetDim(AKdim);
    const int32_t N = param_->N == 0 ? B->GetShape().GetDim(BNdim) : param_->N;

    ppl::kernel::x86::gemm_v2_param_fp32 param;
    param.src_A = A->GetBufferPtr<float>();
    param.src_B = B->GetBufferPtr<float>();
    param.dst_Y = Y->GetBufferPtr<float>();
    param.M = M;
    param.N = N;
    param.K = K;
    param.lda = param_->transA ? M : K;
    param.ldb = param_->transB ? K : N;
    param.ldy = N;
    param.alpha = param_->alpha;
    param.beta = param_->beta;
    param.trans_A = param_->transA;
    param.trans_B = param_->transB;
    param.isa_flag = GetISA();

    if (gemm_fuse_relu_) {
        param.fuse_flag = ppl::kernel::x86::gemm_v2_fuse_flag::RELU;
    } else {
        param.fuse_flag = ppl::kernel::x86::gemm_v2_fuse_flag::NONE;
    }

    param.src_C = nullptr;
    param.c_type = ppl::kernel::x86::gemm_v2_C_type::EMPTY;
    param.ldc = 0;
    if (C != nullptr && !C->GetShape().IsEmpty()) {
        param.src_C = C->GetBufferPtr<float>();
        if (C->GetShape().GetElementsExcludingPadding() == 1) {
            param.c_type = ppl::kernel::x86::gemm_v2_C_type::SCALAR;
        } else if (C->GetShape().GetDimCount() == 1) {
            param.c_type = ppl::kernel::x86::gemm_v2_C_type::VECTOR_W;
        } else if (C->GetShape().GetDimCount() == 2) {
            if (C->GetShape().GetDim(0) == 1) {
                param.c_type = ppl::kernel::x86::gemm_v2_C_type::VECTOR_W;
            } else if (C->GetShape().GetDim(1) == 1) {
                param.c_type = ppl::kernel::x86::gemm_v2_C_type::VECTOR_H;
            } else {
                param.c_type = ppl::kernel::x86::gemm_v2_C_type::MATRIX;
                param.ldc = C->GetShape().GetDim(1);
            }
        }
    }

    auto executor =
        std::unique_ptr<ppl::kernel::x86::gemm_v2_executor_fp32>(ppl::kernel::x86::create_gemm_v2_executor_fp32(param));
    if (!executor) {
        LOG(ERROR) << "cannot create executor.";
        return ppl::common::RC_UNSUPPORTED;
    }

    BufferDesc tmp_buffer_desc;
    uint64_t tmp_buffer_size = executor->get_buffer_bytes();
    auto status = GetX86Device()->AllocTmpBuffer(tmp_buffer_size, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_size << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    BufferDescGuard __tmp_buffer_guard(&tmp_buffer_desc, [this](BufferDesc* buffer) -> void {
        GetX86Device()->FreeTmpBuffer(buffer);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;
    executor->set_temp_buffer(tmp_buffer);

    return executor->execute();
}

}}} // namespace ppl::nn::x86
