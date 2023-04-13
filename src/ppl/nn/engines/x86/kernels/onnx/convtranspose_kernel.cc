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

#include "ppl/nn/engines/x86/kernels/onnx/convtranspose_kernel.h"
#include "ppl/common/destructor.h"
#include "ppl/kernel/x86/fp32/conv_transpose.h"

namespace ppl { namespace nn { namespace x86 {

uint64_t ConvTransposeKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto X = ctx.GetInput<TensorImpl>(0);
    auto W = ctx.GetInput<TensorImpl>(1);

    const int64_t num_output = W->GetShape()->GetDim(1) * param_->group;
    const uint32_t kernel_dims = W->GetShape()->GetDimCount() - 2;

    if (kernel_dims == 2) {
        return ppl::kernel::x86::conv_transpose_2d_ndarray_fp32_get_buffer_bytes(
            GetISA(), X->GetShape(), param_->group, num_output,
            param_->kernel_shape[0], param_->kernel_shape[1],
            param_->strides[0], param_->strides[1],
            param_->pads[0], param_->pads[1]);
    }
    if (kernel_dims == 1) {
        return ppl::kernel::x86::conv_transpose_1d_ndarray_fp32_get_buffer_bytes(
            GetISA(), X->GetShape(), param_->group, num_output,
            param_->kernel_shape[0],
            param_->strides[0],
            param_->pads[0]);
    }
    return 0;
}

ppl::common::RetCode ConvTransposeKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(X, 0);
    PPLNN_X86_REQUIRED_INPUT(W, 1);
    PPLNN_X86_OPTIONAL_INPUT(B, 2);
    PPLNN_X86_REQUIRED_OUTPUT(Y, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [X]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(X);
    PPLNN_X86_DEBUG_TRACE("Input [W]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(W);

    const float* b_data = nullptr;
    if (B) {
        b_data = B->GetBufferPtr<float>();
        PPLNN_X86_DEBUG_TRACE("Input [B]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(B);
    }

    int64_t num_output = W->GetShape()->GetDim(1) * param_->group;
    const int64_t channels = X->GetShape()->GetDim(1);
    const uint32_t kernel_dims = W->GetShape()->GetDimCount() - 2;

    if (kernel_dims == 1) {
        PPLNN_X86_DEBUG_TRACE("kernel_shape: %d\n", param_->kernel_shape[0]);
        PPLNN_X86_DEBUG_TRACE("dilations: %d\n", param_->dilations[0]);
        PPLNN_X86_DEBUG_TRACE("strides: %d\n", param_->strides[0]);
        PPLNN_X86_DEBUG_TRACE("pads: %d %d\n", param_->pads[0], param_->pads[1]);
    }
    if (kernel_dims == 2) {
        PPLNN_X86_DEBUG_TRACE("kernel_shape: %d %d\n", param_->kernel_shape[0], param_->kernel_shape[1]);
        PPLNN_X86_DEBUG_TRACE("dilations: %d %d\n", param_->dilations[0], param_->dilations[1]);
        PPLNN_X86_DEBUG_TRACE("strides: %d %d\n", param_->strides[0], param_->strides[1]);
        PPLNN_X86_DEBUG_TRACE("pads: %d %d %d %d\n", param_->pads[0], param_->pads[1], param_->pads[2], param_->pads[3]);
    }
    PPLNN_X86_DEBUG_TRACE("group: %ld\n", param_->group);
    PPLNN_X86_DEBUG_TRACE("num_output: %ld\n", num_output);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    if (X->GetShape()->GetDataType() != ppl::common::DATATYPE_FLOAT32 ||
        X->GetShape()->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY) {
        LOG(ERROR) << "only support fp32 ndarray now.";
        return ppl::common::RC_UNSUPPORTED;
    }

    for (uint32_t i = 0; i < kernel_dims; ++i) {
        if (param_->pads[i] != param_->pads[i + kernel_dims]) {
            LOG(ERROR) << "only support symmetrical pads.";
            return ppl::common::RC_UNSUPPORTED;
        }
    }

    PPLNN_X86_REALLOC_TENSOR_BUFFER(Y);
    PPLNN_X86_DEBUG_TRACE("Output [Y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);

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

    if (kernel_dims == 1) {
        return kernel::x86::conv_transpose_1d_ndarray_fp32(
            GetISA(), X->GetShape(), Y->GetShape(),
            X->GetBufferPtr<float>(), W->GetBufferPtr<float>(), b_data,
            param_->group, channels, num_output,
            param_->kernel_shape[0], param_->strides[0],
            param_->pads[0], param_->dilations[0],
            tmp_buffer, Y->GetBufferPtr<float>());
    }
    if (kernel_dims == 2) {
        return kernel::x86::conv_transpose_2d_ndarray_fp32(
            GetISA(), X->GetShape(), Y->GetShape(),
            X->GetBufferPtr<float>(), W->GetBufferPtr<float>(), b_data,
            param_->group, channels, num_output,
            param_->kernel_shape[0], param_->kernel_shape[1],
            param_->strides[0], param_->strides[1],
            param_->pads[0], param_->pads[1],
            param_->dilations[0], param_->dilations[1],
            tmp_buffer, Y->GetBufferPtr<float>());
    }
    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
