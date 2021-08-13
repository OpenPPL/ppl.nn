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
#include "ppl/kernel/x86/fp32/conv_transpose.h"

namespace ppl { namespace nn { namespace x86 {

uint64_t ConvTransposeKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto x = ctx.GetInput<TensorImpl>(0);

    const int32_t batch = x->GetShape().GetDim(0);
    const int32_t src_h = x->GetShape().GetDim(2);
    const int32_t src_w = x->GetShape().GetDim(3);
    const int32_t num_outputs = ctx.GetInput<TensorImpl>(1)->GetShape().GetDim(0);
    const int32_t channels = x->GetShape().GetDim(1);

    if (false) {
    }
#ifdef PPL_USE_X86_AVX512
    else if (MayUseISA(ppl::common::ISA_X86_AVX512)) {
        return kernel::x86::conv_transpose_ndarray_fp32_avx512_get_buffer_bytes(
            batch, src_h, src_w, num_outputs, channels, param_->kernel_shape[0], param_->kernel_shape[1],
            param_->strides[0], param_->strides[1], param_->pads[0], param_->pads[1]);
    }
#endif
    else if (MayUseISA(ppl::common::ISA_X86_FMA)) {
        return kernel::x86::conv_transpose_ndarray_fp32_fma_get_buffer_bytes(
            batch, src_h, src_w, num_outputs, channels, param_->kernel_shape[0], param_->kernel_shape[1],
            param_->strides[0], param_->strides[1], param_->pads[0], param_->pads[1]);
    } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
        return kernel::x86::conv_transpose_ndarray_fp32_sse_get_buffer_bytes(
            batch, src_h, src_w, num_outputs, channels, param_->kernel_shape[0], param_->kernel_shape[1],
            param_->strides[0], param_->strides[1], param_->pads[0], param_->pads[1]);
    } else {
        LOG(ERROR) << "unsupported isa: " << GetISA();
    }

    return 0;
}

ppl::common::RetCode ConvTransposeKernel::DoExecute(KernelExecContext* ctx) {
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

    PPLNN_X86_REQUIRED_INPUT(X, 0);
    PPLNN_X86_REQUIRED_INPUT(W, 1);
    PPLNN_X86_OPTIONAL_INPUT(B, 2);
    PPLNN_X86_REQUIRED_OUTPUT(Y, 0);
    const float* b_data = nullptr;

    int32_t num_output = W->GetShape().GetDim(1);
    if (B) {
        b_data = B->GetBufferPtr<float>();
    }

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [X]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(X);
    PPLNN_X86_DEBUG_TRACE("Input [W]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(W);
    if (B) {
        PPLNN_X86_DEBUG_TRACE("Input [B]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(B);
    }
    PPLNN_X86_DEBUG_TRACE("Output [Y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);
    PPLNN_X86_DEBUG_TRACE("kernel_shape: %d %d\n", param_->kernel_shape[0], param_->kernel_shape[1]);
    PPLNN_X86_DEBUG_TRACE("dilations: %d %d\n", param_->dilations[0], param_->dilations[1]);
    PPLNN_X86_DEBUG_TRACE("strides: %d %d\n", param_->strides[0], param_->strides[1]);
    PPLNN_X86_DEBUG_TRACE("pads: %d %d %d %d\n", param_->pads[0], param_->pads[1], param_->pads[2], param_->pads[3]);
    PPLNN_X86_DEBUG_TRACE("group: %ld\n", param_->group);
    PPLNN_X86_DEBUG_TRACE("buffer: %p\n", tmp_buffer);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const int32_t batch = X->GetShape().GetDim(0);
    const int32_t channels = X->GetShape().GetDim(1);
    const int32_t src_h = X->GetShape().GetDim(2);
    const int32_t src_w = X->GetShape().GetDim(3);
    const int32_t dst_h = Y->GetShape().GetDim(2);
    const int32_t dst_w = Y->GetShape().GetDim(3);

    memset(tmp_buffer, 0, CalcTmpBufferSize(*ctx));
    memset(Y->GetBufferPtr<void>(), 0, Y->GetShape().GetBytesIncludingPadding());

    const auto data_format = X->GetShape().GetDataFormat();
    const auto data_type = X->GetShape().GetDataType();

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
            if (false) {
            }
#ifdef PPL_USE_X86_AVX512
            if (MayUseISA(ppl::common::ISA_X86_AVX512)) {
                return kernel::x86::conv_transpose_ndarray_fp32_avx512(
                    X->GetBufferPtr<float>(), W->GetBufferPtr<float>(), b_data, src_h, src_w, dst_h, dst_w, batch,
                    channels, num_output, param_->kernel_shape[0], param_->kernel_shape[1], param_->strides[0],
                    param_->strides[1], param_->pads[0], param_->pads[1], param_->dilations[0], param_->dilations[1],
                    (float*)tmp_buffer, Y->GetBufferPtr<float>());
            }
#endif
            else if (MayUseISA(ppl::common::ISA_X86_FMA)) {
                return kernel::x86::conv_transpose_ndarray_fp32_fma(
                    X->GetBufferPtr<float>(), W->GetBufferPtr<float>(), b_data, src_h, src_w, dst_h, dst_w, batch,
                    channels, num_output, param_->kernel_shape[0], param_->kernel_shape[1], param_->strides[0],
                    param_->strides[1], param_->pads[0], param_->pads[1], param_->dilations[0], param_->dilations[1],
                    (float*)tmp_buffer, Y->GetBufferPtr<float>());
            } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
                return kernel::x86::conv_transpose_ndarray_fp32_sse(
                    X->GetBufferPtr<float>(), W->GetBufferPtr<float>(), b_data, src_h, src_w, dst_h, dst_w, batch,
                    channels, num_output, param_->kernel_shape[0], param_->kernel_shape[1], param_->strides[0],
                    param_->strides[1], param_->pads[0], param_->pads[1], param_->dilations[0], param_->dilations[1],
                    (float*)tmp_buffer, Y->GetBufferPtr<float>());
            } else {
                LOG(ERROR) << "unsupported isa: " << GetISA();
            }
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format) << ".";
        }
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }
    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
