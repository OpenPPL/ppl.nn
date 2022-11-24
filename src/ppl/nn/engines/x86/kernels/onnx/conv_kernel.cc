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

#include "ppl/nn/engines/x86/kernels/onnx/conv_kernel.h"
#include "ppl/common/destructor.h"
#include "ppl/kernel/x86/fp32/conv.h"

namespace ppl { namespace nn { namespace x86 {

uint64_t ConvKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto W = ctx.GetInput<TensorImpl>(1);
    auto Y = ctx.GetOutput<TensorImpl>(0);

    const int32_t channels = W->GetShape()->GetDim(1) * param_->param->group;
    const uint32_t kernel_dims = W->GetShape()->GetDimCount() - 2;

    if (kernel_dims == 1) {
        return ppl::kernel::x86::conv1d_ndarray_fp32_get_buffer_bytes(
            GetISA(), Y->GetShape(), param_->param->group, channels,
            param_->param->kernel_shape[0],
            param_->param->strides[0],
            param_->param->pads[0]);
    }
    if (kernel_dims == 2) {
        return ppl::kernel::x86::conv2d_ndarray_fp32_get_buffer_bytes(
            GetISA(), Y->GetShape(), param_->param->group, channels,
            param_->param->kernel_shape[0], param_->param->kernel_shape[1],
            param_->param->strides[0], param_->param->strides[1],
            param_->param->pads[0], param_->param->pads[1]);
    }
    return 0;
}

ppl::common::RetCode ConvKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(X, 0);
    PPLNN_X86_REQUIRED_INPUT(W, 1);
    PPLNN_X86_REQUIRED_OUTPUT(Y, 0);

    const float* b_data = nullptr;
    const float* sum_src_data = nullptr;
    const TensorShape* sum_src_shape = nullptr;

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());

    PPLNN_X86_DEBUG_TRACE("Input [X]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(X);
    PPLNN_X86_DEBUG_TRACE("Input [W]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(W);

    const int32_t num_output = W->GetShape()->GetDim(0);
    const int32_t channels = W->GetShape()->GetDim(1) * param_->param->group;
    const uint32_t kernel_dims = W->GetShape()->GetDimCount() - 2;

    if (kernel_dims > 2) {
        LOG(ERROR) << "only support conv1d/conv2d now.";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (kernel_dims == 1) {
        PPLNN_X86_DEBUG_TRACE("kernel_shape: %d\n", param_->param->kernel_shape[0]);
        PPLNN_X86_DEBUG_TRACE("dilations: %d\n", param_->param->dilations[0]);
        PPLNN_X86_DEBUG_TRACE("strides: %d\n", param_->param->strides[0]);
        PPLNN_X86_DEBUG_TRACE("pads: %d %d\n", param_->param->pads[0], param_->param->pads[1]);
    }
    if (kernel_dims == 2) {
        PPLNN_X86_DEBUG_TRACE("kernel_shape: %d %d\n", param_->param->kernel_shape[0], param_->param->kernel_shape[1]);
        PPLNN_X86_DEBUG_TRACE("dilations: %d %d\n", param_->param->dilations[0], param_->param->dilations[1]);
        PPLNN_X86_DEBUG_TRACE("strides: %d %d\n", param_->param->strides[0], param_->param->strides[1]);
        PPLNN_X86_DEBUG_TRACE("pads: %d %d %d %d\n", param_->param->pads[0], param_->param->pads[1], param_->param->pads[2], param_->param->pads[3]);
    }
    PPLNN_X86_DEBUG_TRACE("group: %d\n", param_->param->group);
    PPLNN_X86_DEBUG_TRACE("num_output: %d\n", num_output);
    PPLNN_X86_DEBUG_TRACE("bias_term: %d\n", param_->bias_term);
    PPLNN_X86_DEBUG_TRACE("fuse_flag: %d\n", param_->fuse_flag);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    if (param_->bias_term) {
        PPLNN_X86_REQUIRED_INPUT(B, 2);
        PPLNN_X86_DEBUG_TRACE("Input [B]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(B);
        b_data = B->GetBufferPtr<float>();
    }

    if (param_->fuse_flag & ppl::kernel::x86::conv_fuse_flag::SUM) {
        PPLNN_X86_REQUIRED_INPUT(sum_src, param_->bias_term ? 3 : 2);
        PPLNN_X86_DEBUG_TRACE("Input [sum_src]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(sum_src);
        sum_src_data = sum_src->GetBufferPtr<float>();
        sum_src_shape = sum_src->GetShape();
    }

    if (X->GetShape()->GetDataType() != ppl::common::DATATYPE_FLOAT32 ||
        X->GetShape()->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY) {
        LOG(ERROR) << "only support fp32 ndarray now.";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (X->GetShape()->GetDimCount() != 4 || W->GetShape()->GetDimCount() != 4) {
        LOG(ERROR) << "ConvOp only support 4-D Tensor for X & W";
        return ppl::common::RC_UNSUPPORTED;
    }

    for (uint32_t i = 0; i < kernel_dims; ++i) {
        if (param_->param->pads[i] != param_->param->pads[i + 2]) {
            LOG(ERROR) << "ConvOp only support symmetrical pads.";
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
        return kernel::x86::conv1d_ndarray_fp32(
            GetISA(), X->GetShape(), sum_src_shape, Y->GetShape(),
            X->GetBufferPtr<float>(), W->GetBufferPtr<float>(),
            sum_src_data, b_data, param_->param->group, channels, num_output,
            param_->param->kernel_shape[0], param_->param->strides[0],
            param_->param->pads[0], param_->param->dilations[0],
            param_->fuse_flag, tmp_buffer, Y->GetBufferPtr<float>());
    }
    if (kernel_dims == 2) {
        return kernel::x86::conv2d_ndarray_fp32(
            GetISA(), X->GetShape(), sum_src_shape, Y->GetShape(),
            X->GetBufferPtr<float>(), W->GetBufferPtr<float>(),
            sum_src_data, b_data, param_->param->group, channels, num_output,
            param_->param->kernel_shape[0], param_->param->kernel_shape[1],
            param_->param->strides[0], param_->param->strides[1],
            param_->param->pads[0], param_->param->pads[1],
            param_->param->dilations[0], param_->param->dilations[1],
            param_->fuse_flag, tmp_buffer, Y->GetBufferPtr<float>());
    }
    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
