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

#include "ppl/nn/engines/riscv/kernels/onnx/conv_transpose_kernel.h"
#include "ppl/common/destructor.h"
#include "ppl/kernel/riscv/fp32/conv_transpose.h"
#include "ppl/kernel/riscv/fp16/conv_transpose.h"

namespace ppl { namespace nn { namespace riscv {

uint64_t ConvTransposeKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto x = ctx.GetInput<TensorImpl>(0);

    const int32_t batch = x->GetShape()->GetDim(0);
    const int32_t src_h = x->GetShape()->GetDim(2);
    const int32_t src_w = x->GetShape()->GetDim(3);
    const int32_t num_outputs = ctx.GetInput<TensorImpl>(1)->GetShape()->GetDim(1);
    const int32_t channels = x->GetShape()->GetDim(1);

    return kernel::riscv::conv_transpose_n4cx_get_buffer_bytes_fp32_vec128(
        batch, src_h, src_w, num_outputs, channels, param_->kernel_shape[0], param_->kernel_shape[1],
        param_->strides[0], param_->strides[1], param_->pads[0], param_->pads[1]);
}

ppl::common::RetCode ConvTransposeKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_RISCV_REQUIRED_INPUT(X, 0);
    PPLNN_RISCV_REQUIRED_INPUT(W, 1);
    PPLNN_RISCV_OPTIONAL_INPUT(B, 2);
    PPLNN_RISCV_REQUIRED_OUTPUT(Y, 0);

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [X]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(X);
    PPLNN_RISCV_DEBUG_TRACE("Input [W]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(W);
    if (B) {
        PPLNN_RISCV_DEBUG_TRACE("Input [B]:\n");
        PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(B);
    }

    PPLNN_RISCV_DEBUG_TRACE("kernel_shape: %d %d\n", param_->kernel_shape[0], param_->kernel_shape[1]);
    PPLNN_RISCV_DEBUG_TRACE("dilations: %d %d\n", param_->dilations[0], param_->dilations[1]);
    PPLNN_RISCV_DEBUG_TRACE("strides: %d %d\n", param_->strides[0], param_->strides[1]);
    PPLNN_RISCV_DEBUG_TRACE("pads: %d %d %d %d\n", param_->pads[0], param_->pads[1], param_->pads[2], param_->pads[3]);
    PPLNN_RISCV_DEBUG_TRACE("group: %ld\n", param_->group);

    const int32_t batch = X->GetShape()->GetDim(0);
    const int32_t channels = X->GetShape()->GetDim(1);
    const int32_t src_h = X->GetShape()->GetDim(2);
    const int32_t src_w = X->GetShape()->GetDim(3);
    const int32_t dst_h = Y->GetShape()->GetDim(2);
    const int32_t dst_w = Y->GetShape()->GetDim(3);

    // PPLNN_RISCV_REALLOC_TENSOR_BUFFER(Y);
    PPLNN_RISCV_DEBUG_TRACE("Output [Y]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(Y);

    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_size = CalcTmpBufferSize(*ctx);
    auto status = GetRiscvDevice()->AllocTmpBuffer(tmp_buffer_size, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_size << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    ppl::common::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
        GetRiscvDevice()->FreeTmpBuffer(&tmp_buffer_desc);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;
    PPLNN_RISCV_DEBUG_TRACE("buffer: %p\n", tmp_buffer);

    const auto data_format = X->GetShape()->GetDataFormat();
    const auto data_type = X->GetShape()->GetDataType();
    int32_t num_output = W->GetShape()->GetDim(1);

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        const float* w_data = (float*)conv_transpose_param_->weight.get();
        const float* b_data = (float*)conv_transpose_param_->bias.get();

        if (data_format == ppl::common::DATAFORMAT_N4CX) {
            return kernel::riscv::conv_transpose_n4cx_fp32_vec128(
                X->GetBufferPtr<float>(), w_data, b_data, src_h, src_w, dst_h, dst_w, batch, channels, num_output,
                param_->kernel_shape[0], param_->kernel_shape[1], param_->strides[0], param_->strides[1],
                param_->pads[0], param_->pads[1], param_->dilations[0], param_->dilations[1], (float*)tmp_buffer,
                Y->GetBufferPtr<float>());
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format) << ".";
        }
    } else if (data_type == ppl::common::DATATYPE_FLOAT16) {
        const __fp16* w_data = (__fp16*)conv_transpose_param_->weight.get();
        const __fp16* b_data = (__fp16*)conv_transpose_param_->bias.get();

        if (data_format == ppl::common::DATAFORMAT_N8CX) {
            return kernel::riscv::conv_transpose_n8cx_fp16_vec128(
                X->GetBufferPtr<__fp16>(), w_data, b_data, src_h, src_w, dst_h, dst_w, batch, channels, num_output,
                param_->kernel_shape[0], param_->kernel_shape[1], param_->strides[0], param_->strides[1],
                param_->pads[0], param_->pads[1], param_->dilations[0], param_->dilations[1], (__fp16*)tmp_buffer,
                Y->GetBufferPtr<__fp16>());
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format) << ".";
        }
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }
    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::riscv
