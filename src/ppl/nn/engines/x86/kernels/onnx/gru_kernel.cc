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

#include "ppl/nn/engines/x86/kernels/onnx/gru_kernel.h"
#include "ppl/common/destructor.h"
#include "ppl/kernel/x86/fp32/gru.h"

namespace ppl { namespace nn { namespace x86 {

bool GRUKernel::CanDoExecute(const KernelExecContext& ctx) const {
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

uint64_t GRUKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto X = ctx.GetInput<TensorImpl>(0);
    const bool has_Y = ctx.GetOutputCount() > 0 && ctx.GetOutput<TensorImpl>(0);
    const bool has_Y_h = ctx.GetOutputCount() > 1 && ctx.GetOutput<TensorImpl>(1);

    return kernel::x86::gru_fp32_get_buffer_bytes(X->GetShape(), direction_, param_->param->hidden_size, has_Y,
                                                  has_Y_h);
}

ppl::common::RetCode GRUKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(X, 0);
    PPLNN_X86_REQUIRED_INPUT(W, 1);
    PPLNN_X86_REQUIRED_INPUT(R, 2);
    PPLNN_X86_OPTIONAL_INPUT(B, 3);
    PPLNN_X86_OPTIONAL_INPUT(sequence_lens, 4);
    PPLNN_X86_OPTIONAL_INPUT(initial_h, 5);
    PPLNN_X86_OPTIONAL_OUTPUT(Y, 0);
    PPLNN_X86_OPTIONAL_OUTPUT(Y_h, 1);

    const float* B_data = nullptr;
    const int32_t* sequence_lens_data = nullptr;
    const float* initial_h_data = nullptr;
    float* Y_data = nullptr;
    float* Y_h_data = nullptr;

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
    PPLNN_X86_DEBUG_TRACE("linear_before_reset: %d\n", param_->param->linear_before_reset); // default 1 in torch
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

    const float* real_W[2] = {param_->packed_W[0], param_->packed_W[1]};
    const float* real_Rzr[2] = {param_->packed_Rzr[0], param_->packed_Rzr[1]};
    const float* real_Rh[2] = {param_->packed_Rh[0], param_->packed_Rh[1]};
    bool has_packed_W = real_W[0] != nullptr;
    bool has_packed_Rzr = real_Rzr[0] != nullptr;
    bool has_packed_Rh = real_Rh[0] != nullptr;
    if (!real_W[0])
        real_W[0] = W->GetBufferPtr<const float>();
    if (!real_W[1])
        real_W[1] = W->GetBufferPtr<const float>() + W->GetShape()->GetDim(1) * W->GetShape()->GetDim(2);
    if (!real_Rzr[0])
        real_Rzr[0] = R->GetBufferPtr<const float>();
    if (!real_Rh[0])
        real_Rh[0] = R->GetBufferPtr<const float>() + R->GetShape()->GetDim(1) / 3 * 2 * R->GetShape()->GetDim(2);
    if (!real_Rzr[1])
        real_Rzr[1] = R->GetBufferPtr<const float>() + R->GetShape()->GetDim(1) * R->GetShape()->GetDim(2);
    if (!real_Rh[1])
        real_Rh[1] = R->GetBufferPtr<const float>() + R->GetShape()->GetDim(1) / 3 * 5 * R->GetShape()->GetDim(2);
    if (data_type == ppl::common::DATATYPE_FLOAT32 && data_format == ppl::common::DATAFORMAT_NDARRAY) {
        return kernel::x86::gru_fp32(GetISA(), X->GetShape(), X->GetBufferPtr<const float>(), real_W, real_Rzr, real_Rh,
                                     B_data, sequence_lens_data, initial_h_data, direction_, param_->param->hidden_size,
                                     has_packed_W, has_packed_Rzr, has_packed_Rh, tmp_buffer, Y_data, Y_h_data);
    } else {
        LOG(ERROR) << "only support fp32 ndarray now.";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
