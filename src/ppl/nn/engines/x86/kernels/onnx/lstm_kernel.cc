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

#include "ppl/nn/engines/x86/kernels/onnx/lstm_kernel.h"
#include "ppl/kernel/x86/fp32/lstm.h"

namespace ppl { namespace nn { namespace x86 {

bool LSTMKernel::CanDoExecute(const KernelExecContext& ctx) const {
    if (ctx.GetInputCount() < 3) {
        return false;
    }

    auto X = ctx.GetInput<TensorImpl>(0);
    auto W = ctx.GetInput<TensorImpl>(1);
    auto R = ctx.GetInput<TensorImpl>(2);

    if (!X || !W || !R) {
        return false;
    }

    return true;
}

uint64_t LSTMKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto X = ctx.GetInput<TensorImpl>(0);
    const bool has_Y = ctx.GetOutputCount() > 0 && ctx.GetOutput<TensorImpl>(0);
    const bool has_Y_h = ctx.GetOutputCount() > 1 && ctx.GetOutput<TensorImpl>(1);
    const bool has_Y_c = ctx.GetOutputCount() > 2 && ctx.GetOutput<TensorImpl>(2);
    if (MayUseISA(ppl::common::ISA_X86_FMA)) {
        return kernel::x86::lstm_fp32_fma_get_buffer_bytes(
            &X->GetShape(), direction_, param_->hidden_size, has_Y, has_Y_h, has_Y_c);
    } else {
        return kernel::x86::lstm_ref_fp32_get_buffer_bytes(
            &X->GetShape(), direction_, param_->hidden_size, has_Y, has_Y_h, has_Y_c);
    }
}

ppl::common::RetCode LSTMKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(X, 0);
    PPLNN_X86_REQUIRED_INPUT(W, 1);
    PPLNN_X86_REQUIRED_INPUT(R, 2);
    PPLNN_X86_OPTIONAL_INPUT(B, 3);
    PPLNN_X86_OPTIONAL_INPUT(sequence_lens, 4);
    PPLNN_X86_OPTIONAL_INPUT(initial_h, 5);
    PPLNN_X86_OPTIONAL_INPUT(initial_c, 6);
    PPLNN_X86_OPTIONAL_INPUT(P, 7);
    PPLNN_X86_OPTIONAL_OUTPUT(Y, 0);
    PPLNN_X86_OPTIONAL_OUTPUT(Y_h, 1);
    PPLNN_X86_OPTIONAL_OUTPUT(Y_c, 2);

    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_size = CalcTmpBufferSize(*ctx);
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

    const float *B_data = nullptr;
    const int32_t *sequence_lens_data = nullptr;
    const float *initial_h_data = nullptr;
    const float *initial_c_data = nullptr;
    const float *P_data = nullptr;
    float *Y_data = nullptr;
    float *Y_h_data = nullptr;
    float *Y_c_data = nullptr;

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [X]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(X);
    PPLNN_X86_DEBUG_TRACE("Input [W]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(W);
    PPLNN_X86_DEBUG_TRACE("Input [R]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(R);
    if (B) {
        PPLNN_X86_DEBUG_TRACE("Input [B]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(B);
        B_data = B->GetBufferPtr<const float>();
    }
    if (sequence_lens) {
        PPLNN_X86_DEBUG_TRACE("Input [sequence_lens]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(sequence_lens);
        sequence_lens_data = sequence_lens->GetBufferPtr<const int32_t>();
    }
    if (initial_h) {
        PPLNN_X86_DEBUG_TRACE("Input [initial_h]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(initial_h);
        initial_h_data = initial_h->GetBufferPtr<const float>();
    }
    if (initial_c) {
        PPLNN_X86_DEBUG_TRACE("Input [initial_c]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(initial_c);
        initial_c_data = initial_c->GetBufferPtr<const float>();
    }
    if (P) {
        PPLNN_X86_DEBUG_TRACE("Input [P]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(P);
        P_data = P->GetBufferPtr<const float>();
    }
    if (Y) {
        PPLNN_X86_DEBUG_TRACE("Output [Y]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);
        Y_data = Y->GetBufferPtr<float>();
    }
    if (Y_h) {
        PPLNN_X86_DEBUG_TRACE("Output [Y_h]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y_h);
        Y_h_data = Y_h->GetBufferPtr<float>();
    }
    if (Y_c) {
        PPLNN_X86_DEBUG_TRACE("Output [Y_c]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y_c);
        Y_c_data = Y_c->GetBufferPtr<float>();
    }
    PPLNN_X86_DEBUG_TRACE("activation_alpha(%lu):\n", param_->activation_alpha.size());
    for (size_t i = 0; i < param_->activation_alpha.size(); ++i) {
        PPLNN_X86_DEBUG_TRACE("\t%f\n", param_->activation_alpha[i]);
    }
    PPLNN_X86_DEBUG_TRACE("activation_beta(%lu):\n", param_->activation_beta.size());
    for (size_t i = 0; i < param_->activation_beta.size(); ++i) {
        PPLNN_X86_DEBUG_TRACE("\t%f\n", param_->activation_beta[i]);
    }
    PPLNN_X86_DEBUG_TRACE("activations(%lu):\n", param_->activations.size());
    for (size_t i = 0; i < param_->activations.size(); ++i) {
        PPLNN_X86_DEBUG_TRACE("\t%d\n", param_->activations[i]);
    }
    PPLNN_X86_DEBUG_TRACE("clip: %f\n", param_->clip);
    PPLNN_X86_DEBUG_TRACE("direction: %d\n", param_->direction);
    PPLNN_X86_DEBUG_TRACE("hidden_size: %d\n", param_->hidden_size);
    PPLNN_X86_DEBUG_TRACE("input_forget: %d\n", param_->input_forget);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = X->GetShape().GetDataType();
    const auto data_format = X->GetShape().GetDataFormat();

    if (data_type == ppl::common::DATATYPE_FLOAT32 && data_format == ppl::common::DATAFORMAT_NDARRAY) {
        if (MayUseISA(ppl::common::ISA_X86_FMA)) {
            return kernel::x86::lstm_fp32_fma(
                &X->GetShape(), X->GetBufferPtr<const float>(),
                W->GetBufferPtr<const float>(), R->GetBufferPtr<const float>(),
                P_data, B_data, sequence_lens_data, initial_h_data, initial_c_data,
                direction_, param_->hidden_size, tmp_buffer, Y_data, Y_h_data, Y_c_data);
        } else {
            return kernel::x86::lstm_ref_fp32(
                &X->GetShape(), X->GetBufferPtr<const float>(),
                W->GetBufferPtr<const float>(), R->GetBufferPtr<const float>(),
                P_data, B_data, sequence_lens_data, initial_h_data, initial_c_data,
                direction_, param_->hidden_size, tmp_buffer, Y_data, Y_h_data, Y_c_data);
        }
    } else {
        LOG(ERROR) << "only support fp32 ndarray now.";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
