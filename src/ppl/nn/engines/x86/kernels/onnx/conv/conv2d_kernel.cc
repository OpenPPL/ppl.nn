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

#include "ppl/nn/engines/x86/kernels/onnx/conv/conv2d_kernel.h"

#define CASE_STRING_FMT() "g%ld_mb%d_ic%ldih%diw%d_oc%ldoh%dow%d_kh%ldkw%ldsh%ldsw%ldph%ldpw%lddh%lddw%ld_n%s"

namespace ppl { namespace nn { namespace x86 {

uint64_t Conv2dKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    return use_fallback_ ? fallback_executor_->cal_temp_buffer_size() : executor_->cal_temp_buffer_size();
}

ppl::common::RetCode Conv2dKernel::DoExecute(KernelExecContext* ctx) {
    TensorImpl* X = ctx->GetInput<TensorImpl>(0);
    TensorImpl* Y = ctx->GetOutput<TensorImpl>(0);

    if (param_->infer_fallback_func) {
        use_fallback_ = param_->infer_fallback_func(X, Y, &param_->param);
    }

    auto cur_executor = use_fallback_ ? fallback_executor_ : executor_;

    cur_executor->set_src_shape(&X->GetShape());
    cur_executor->set_src(X->GetBufferPtr<float>());

    cur_executor->set_dst_shape(&Y->GetShape());
    cur_executor->set_dst(Y->GetBufferPtr<float>());

    TensorImpl* sum_src = nullptr;
    if (cur_executor->conv_param()->fuse_flag & ppl::kernel::x86::conv_fuse_flag::SUM) {
        sum_src = ctx->GetInput<TensorImpl>(ctx->GetInputCount() - 1);
        cur_executor->set_sum_src_shape(&sum_src->GetShape());
        cur_executor->set_sum_src(sum_src->GetBufferPtr<float>());
    }

    ppl::common::RetCode rc;
    rc = cur_executor->prepare();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "Prepare failed: " << ppl::common::GetRetCodeStr(rc);
        return rc;
    }

#if DUMP_CONV
    fprintf(stderr, CASE_STRING_FMT() "\n", cur_executor->conv_param()->group, X->GetShape().GetDim(0),
            cur_executor->conv_param()->channels, X->GetShape().GetDim(2), X->GetShape().GetDim(3),
            cur_executor->conv_param()->num_output, Y->GetShape().GetDim(2), Y->GetShape().GetDim(3),
            cur_executor->conv_param()->kernel_h, cur_executor->conv_param()->kernel_w,
            cur_executor->conv_param()->stride_h, cur_executor->conv_param()->stride_w,
            cur_executor->conv_param()->pad_h, cur_executor->conv_param()->pad_w,
            cur_executor->conv_param()->dilation_h - 1, cur_executor->conv_param()->dilation_w - 1, GetName().c_str());
#endif

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

    cur_executor->set_temp_buffer(tmp_buffer);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [X]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(X);
    if (sum_src) {
        PPLNN_X86_DEBUG_TRACE("Input [sum_src]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(sum_src);
    }
    PPLNN_X86_DEBUG_TRACE("Output [Y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);
    PPLNN_X86_DEBUG_TRACE("kernel_shape: %ld %ld\n", cur_executor->conv_param()->kernel_h,
                          cur_executor->conv_param()->kernel_w);
    PPLNN_X86_DEBUG_TRACE("dilations: %ld %ld\n", cur_executor->conv_param()->dilation_h,
                          cur_executor->conv_param()->dilation_w);
    PPLNN_X86_DEBUG_TRACE("strides: %ld %ld\n", cur_executor->conv_param()->stride_h,
                          cur_executor->conv_param()->stride_w);
    PPLNN_X86_DEBUG_TRACE("pads: %ld %ld\n", cur_executor->conv_param()->pad_h, cur_executor->conv_param()->pad_w);
    PPLNN_X86_DEBUG_TRACE("group: %ld\n", cur_executor->conv_param()->group);
    PPLNN_X86_DEBUG_TRACE("channels: %ld\n", cur_executor->conv_param()->channels);
    PPLNN_X86_DEBUG_TRACE("num_output: %ld\n", cur_executor->conv_param()->num_output);
    PPLNN_X86_DEBUG_TRACE("buffer: %p\n", tmp_buffer);
    PPLNN_X86_DEBUG_TRACE("fuse_flag: %ld\n", cur_executor->conv_param()->fuse_flag);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    rc = cur_executor->execute();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "Execute failed: " << ppl::common::GetRetCodeStr(rc);
        return rc;
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::x86
