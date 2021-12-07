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

#include <inttypes.h>

#include "ppl/nn/engines/x86/kernels/ppl/post_depthwise_conv2d_kernel.h"

#define CASE_STRING_FMT() \
    "g%" PRId64 \
    "_mb%" PRId64 \
    "_ic%" PRId64 "ih%" PRId64 "iw%" PRId64 \
    "_oc%" PRId64 "oh%" PRId64 "ow%" PRId64 \
    "_kh%" PRId64 "kw%" PRId64 "sh%" PRId64 "sw%" PRId64 "ph%" PRId64 "pw%" PRId64 "dh%" PRId64 "dw%" PRId64 \
    "_kh%" PRId64 "kw%" PRId64 "sh%" PRId64 "sw%" PRId64 "ph%" PRId64 "pw%" PRId64 "dh%" PRId64 "dw%" PRId64 \
    "_n%s"

namespace ppl { namespace nn { namespace x86 {

uint64_t PostDepthwiseConv2dKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    return executor_->cal_temp_buffer_size();
}

ppl::common::RetCode PostDepthwiseConv2dKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(X, 0);
    PPLNN_X86_REQUIRED_OUTPUT(Y, 0);

    executor_->set_src_shape(&X->GetShape());
    executor_->set_src(X->GetBufferPtr<float>());

    executor_->set_dst_shape(&Y->GetShape());
    executor_->set_dst(Y->GetBufferPtr<float>());

    ppl::common::RetCode rc;
    rc = executor_->prepare();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "Prepare failed: " << ppl::common::GetRetCodeStr(rc);
        return rc;
    }

#if DUMP_CONV
    fprintf(stderr, CASE_STRING_FMT() "\n", executor_->conv2d_executor()->conv_param()->group, X->GetShape().GetDim(0),
            executor_->conv2d_executor()->conv_param()->channels, X->GetShape().GetDim(2), X->GetShape().GetDim(3),
            executor_->conv2d_executor()->conv_param()->num_output, Y->GetShape().GetDim(2), Y->GetShape().GetDim(3),
            executor_->conv2d_executor()->conv_param()->kernel_h, executor_->conv2d_executor()->conv_param()->kernel_w,
            executor_->conv2d_executor()->conv_param()->stride_h, executor_->conv2d_executor()->conv_param()->stride_w,
            executor_->conv2d_executor()->conv_param()->pad_h, executor_->conv2d_executor()->conv_param()->pad_w,
            executor_->conv2d_executor()->conv_param()->dilation_h - 1, executor_->conv2d_executor()->conv_param()->dilation_w - 1,
            executor_->depthwise_conv2d_executor()->conv_param()->kernel_h, executor_->depthwise_conv2d_executor()->conv_param()->kernel_w,
            executor_->depthwise_conv2d_executor()->conv_param()->stride_h, executor_->depthwise_conv2d_executor()->conv_param()->stride_w,
            executor_->depthwise_conv2d_executor()->conv_param()->pad_h, executor_->depthwise_conv2d_executor()->conv_param()->pad_w,
            executor_->depthwise_conv2d_executor()->conv_param()->dilation_h - 1, executor_->depthwise_conv2d_executor()->conv_param()->dilation_w - 1,
            GetName().c_str());
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

    executor_->set_temp_buffer(tmp_buffer);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [X]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(X);
    PPLNN_X86_DEBUG_TRACE("Output [Y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);
    PPLNN_X86_DEBUG_TRACE("kernel_shape: %ld %ld, %ld %ld\n",
        executor_->conv2d_executor()->conv_param()->kernel_h,
        executor_->conv2d_executor()->conv_param()->kernel_w,
        executor_->depthwise_conv2d_executor()->conv_param()->kernel_h,
        executor_->depthwise_conv2d_executor()->conv_param()->kernel_w);
    PPLNN_X86_DEBUG_TRACE("dilations: %ld %ld, %ld %ld\n",
        executor_->conv2d_executor()->conv_param()->dilation_h,
        executor_->conv2d_executor()->conv_param()->dilation_w,
        executor_->depthwise_conv2d_executor()->conv_param()->dilation_h,
        executor_->depthwise_conv2d_executor()->conv_param()->dilation_w);
    PPLNN_X86_DEBUG_TRACE("strides: %ld %ld, %ld %ld\n",
        executor_->conv2d_executor()->conv_param()->stride_h,
        executor_->conv2d_executor()->conv_param()->stride_w,
        executor_->depthwise_conv2d_executor()->conv_param()->stride_h,
        executor_->depthwise_conv2d_executor()->conv_param()->stride_w);
    PPLNN_X86_DEBUG_TRACE("pads: %ld %ld, %ld %ld\n",
        executor_->conv2d_executor()->conv_param()->pad_h,
        executor_->conv2d_executor()->conv_param()->pad_w,
        executor_->depthwise_conv2d_executor()->conv_param()->pad_h,
        executor_->depthwise_conv2d_executor()->conv_param()->pad_w);
    PPLNN_X86_DEBUG_TRACE("group: %ld\n", executor_->conv2d_executor()->conv_param()->group);
    PPLNN_X86_DEBUG_TRACE("channels: %ld\n", executor_->conv2d_executor()->conv_param()->channels);
    PPLNN_X86_DEBUG_TRACE("num_output: %ld\n", executor_->conv2d_executor()->conv_param()->num_output);
    PPLNN_X86_DEBUG_TRACE("buffer: %p\n", tmp_buffer);
    PPLNN_X86_DEBUG_TRACE("fuse_flag: %ld, %ld\n",
        executor_->conv2d_executor()->conv_param()->fuse_flag,
        executor_->depthwise_conv2d_executor()->conv_param()->fuse_flag);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    rc = executor_->execute();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "Execute failed: " << ppl::common::GetRetCodeStr(rc);
        return rc;
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::x86
