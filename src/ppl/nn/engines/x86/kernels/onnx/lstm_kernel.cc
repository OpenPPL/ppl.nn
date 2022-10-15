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
#include "ppl/common/destructor.h"
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
    return kernel::x86::lstm_fp32_get_buffer_bytes(X->GetShape(), direction_, param_->param->hidden_size, has_Y,
                                                   has_Y_h, has_Y_c);
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

    const float* B_data = nullptr;
    const int32_t* sequence_lens_data = nullptr;
    const float* initial_h_data = nullptr;
    const float* initial_c_data = nullptr;
    const float* P_data = nullptr;
    float* Y_data = nullptr;
    float* Y_h_data = nullptr;
    float* Y_c_data = nullptr;

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
    PPLNN_X86_DEBUG_TRACE("activation_alpha(%lu):\n", param_->param->activation_alpha.size());
    for (size_t i = 0; i < param_->param->activation_alpha.size(); ++i) {
        PPLNN_X86_DEBUG_TRACE("\t%f\n", param_->param->activation_alpha[i]);
    }
    PPLNN_X86_DEBUG_TRACE("activation_beta(%lu):\n", param_->param->activation_beta.size());
    for (size_t i = 0; i < param_->param->activation_beta.size(); ++i) {
        PPLNN_X86_DEBUG_TRACE("\t%f\n", param_->param->activation_beta[i]);
    }
    PPLNN_X86_DEBUG_TRACE("activations(%lu):\n", param_->param->activations.size());
    for (size_t i = 0; i < param_->param->activations.size(); ++i) {
        PPLNN_X86_DEBUG_TRACE("\t%d\n", param_->param->activations[i]);
    }
    PPLNN_X86_DEBUG_TRACE("clip: %f\n", param_->param->clip);
    PPLNN_X86_DEBUG_TRACE("direction: %d\n", param_->param->direction);
    PPLNN_X86_DEBUG_TRACE("hidden_size: %d\n", param_->param->hidden_size);
    PPLNN_X86_DEBUG_TRACE("input_forget: %d\n", param_->param->input_forget);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    if (Y) {
        PPLNN_X86_REALLOC_TENSOR_BUFFER(Y);
        PPLNN_X86_DEBUG_TRACE("Output [Y]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);
        Y_data = Y->GetBufferPtr<float>();
    }
    if (Y_h) {
        PPLNN_X86_REALLOC_TENSOR_BUFFER(Y_h);
        PPLNN_X86_DEBUG_TRACE("Output [Y_h]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y_h);
        Y_h_data = Y_h->GetBufferPtr<float>();
    }
    if (Y_c) {
        PPLNN_X86_REALLOC_TENSOR_BUFFER(Y_c);
        PPLNN_X86_DEBUG_TRACE("Output [Y_c]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y_c);
        Y_c_data = Y_c->GetBufferPtr<float>();
    }

    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_size = CalcTmpBufferSize(*ctx);
    auto status = GetX86Device()->AllocTmpBuffer(tmp_buffer_size, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_size << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    ppl::common::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
        GetX86Device()->FreeTmpBuffer(&tmp_buffer_desc);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;
    PPLNN_X86_DEBUG_TRACE("buffer: %p\n", tmp_buffer);

    const auto data_type = X->GetShape()->GetDataType();
    const auto data_format = X->GetShape()->GetDataFormat();

    const float* real_w[2] = {param_->packed_w[0], param_->packed_w[1]};
    const float* real_r[2] = {param_->packed_r[0], param_->packed_r[1]};
    bool has_packed_w = real_w[0] != nullptr;
    bool has_packed_r = real_r[0] != nullptr;

    if (!real_w[0])
        real_w[0] = W->GetBufferPtr<const float>();
    if (!real_w[1])
        real_w[1] = W->GetBufferPtr<const float>() + W->GetShape()->GetDim(1) * W->GetShape()->GetDim(2);
    if (!real_r[0])
        real_r[0] = R->GetBufferPtr<const float>();
    if (!real_r[1])
        real_r[1] = R->GetBufferPtr<const float>() + R->GetShape()->GetDim(1) * R->GetShape()->GetDim(2);

    if (data_type == ppl::common::DATATYPE_FLOAT32 && data_format == ppl::common::DATAFORMAT_NDARRAY) {
        return kernel::x86::lstm_fp32(GetISA(), X->GetShape(), X->GetBufferPtr<const float>(), real_w, real_r, P_data,
                                      B_data, sequence_lens_data, initial_h_data, initial_c_data, direction_,
                                      param_->param->hidden_size, has_packed_w, has_packed_r, tmp_buffer, Y_data,
                                      Y_h_data, Y_c_data);

    } else {
        LOG(ERROR) << "only support fp32 ndarray now.";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
