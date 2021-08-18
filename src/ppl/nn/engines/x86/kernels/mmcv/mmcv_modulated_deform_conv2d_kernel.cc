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

#include "ppl/nn/engines/x86/kernels/mmcv/mmcv_modulated_deform_conv2d_kernel.h"

#include "ppl/kernel/x86/fp32/deform_conv2d.h"

namespace ppl { namespace nn { namespace x86 {

uint64_t MMCVModulatedDeformConv2dKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto weight = ctx.GetInput<TensorImpl>(3);
    auto output = ctx.GetOutput<TensorImpl>(0);
    auto channels = weight->GetShape().GetDim(1) * param_->groups;
    return ppl::kernel::x86::deform_conv2d_ref_fp32_get_buffer_bytes(
        output->GetShape().GetDim(2), output->GetShape().GetDim(3), param_->groups,
        channels, weight->GetShape().GetDim(2), weight->GetShape().GetDim(3));
}

ppl::common::RetCode MMCVModulatedDeformConv2dKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(input, 0);
    PPLNN_X86_REQUIRED_INPUT(offset, 1);
    PPLNN_X86_REQUIRED_INPUT(mask, 2);
    PPLNN_X86_REQUIRED_INPUT(weight, 3);
    PPLNN_X86_OPTIONAL_INPUT(bias, 4);
    PPLNN_X86_REQUIRED_OUTPUT(output, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [input]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_X86_DEBUG_TRACE("Input [offset]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(offset);
    PPLNN_X86_DEBUG_TRACE("Input [mask]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(mask);
    PPLNN_X86_DEBUG_TRACE("Input [weight]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(weight);
    PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_X86_DEBUG_TRACE("stride: %ld, %ld\n", param_->stride[0], param_->stride[1]);
    PPLNN_X86_DEBUG_TRACE("padding: %ld, %ld\n", param_->padding[0], param_->padding[1]);
    PPLNN_X86_DEBUG_TRACE("dilation: %ld, %ld\n", param_->dilation[0], param_->dilation[1]);
    PPLNN_X86_DEBUG_TRACE("groups: %ld\n", param_->groups);
    PPLNN_X86_DEBUG_TRACE("deform_groups: %ld\n", param_->deform_groups);

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

    const int64_t num_output = weight->GetShape().GetDim(0);
    const int64_t channels = weight->GetShape().GetDim(1) * param_->groups;
    const int64_t kernel_h = weight->GetShape().GetDim(2);
    const int64_t kernel_w = weight->GetShape().GetDim(3);

    const float *b_data = nullptr;
    if (bias) {
        b_data = bias->GetBufferPtr<const float>();
    }

    return ppl::kernel::x86::deform_conv2d_ref_fp32(
        &input->GetShape(), &output->GetShape(),
        input->GetBufferPtr<const float>(), offset->GetBufferPtr<const float>(),
        mask->GetBufferPtr<const float>(), weight->GetBufferPtr<const float>(), b_data,
        param_->groups, param_->deform_groups, channels, num_output,
        kernel_h, kernel_w, param_->stride[0], param_->stride[1],
        param_->padding[0], param_->padding[1], param_->dilation[0], param_->dilation[1],
        tmp_buffer, output->GetBufferPtr<float>());
}

}}} // namespace ppl::nn::x86
