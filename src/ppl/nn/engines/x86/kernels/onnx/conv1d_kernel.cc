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
#include "ppl/common/destructor.h"
#include "ppl/nn/engines/x86/kernels/onnx/conv1d_kernel.h"

#define CASE_STRING_FMT() \
    "g%" PRId64 \
    "_mb%" PRId64 \
    "_ic%" PRId64 "ih%" PRId64 "iw%" PRId64 \
    "_oc%" PRId64 "oh%" PRId64 "ow%" PRId64 \
    "_kh%" PRId64 "kw%" PRId64 "sh%" PRId64 "sw%" PRId64 "ph%" PRId64 "pw%" PRId64 "dh%" PRId64 "dw%" PRId64 \
    "_n%s"

namespace ppl { namespace nn { namespace x86 {

uint64_t Conv1dKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    return use_fallback_ ? fallback_executor_->cal_temp_buffer_size() : executor_->cal_temp_buffer_size();
}

ppl::common::RetCode Conv1dKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(X, 0);
    PPLNN_X86_REQUIRED_OUTPUT(Y, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [X]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(X);

    if (param_->infer_fallback_func) {
        use_fallback_ = param_->infer_fallback_func(X, Y, &param_->param);
    }

    auto cur_executor = use_fallback_ ? fallback_executor_ : executor_;

    PPLNN_X86_DEBUG_TRACE("kernel_shape: %ld\n", cur_executor->conv_param()->kernel_w);
    PPLNN_X86_DEBUG_TRACE("dilations: %ld\n", cur_executor->conv_param()->dilation_w);
    PPLNN_X86_DEBUG_TRACE("strides: %ld\n", cur_executor->conv_param()->stride_w);
    PPLNN_X86_DEBUG_TRACE("pads: %ld\n", cur_executor->conv_param()->pad_w);
    PPLNN_X86_DEBUG_TRACE("group: %ld\n", cur_executor->conv_param()->group);
    PPLNN_X86_DEBUG_TRACE("channels: %ld\n", cur_executor->conv_param()->channels);
    PPLNN_X86_DEBUG_TRACE("num_output: %ld\n", cur_executor->conv_param()->num_output);
    PPLNN_X86_DEBUG_TRACE("fuse_flag: %ld\n", cur_executor->conv_param()->fuse_flag);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    TensorShape X_shape = *X->GetShape();
    TensorShape Y_shape = *Y->GetShape();
    X_shape.Reshape({X_shape.GetDim(0), X_shape.GetDim(1), 1, X_shape.GetDim(2)});
    Y_shape.Reshape({Y_shape.GetDim(0), Y_shape.GetDim(1), 1, Y_shape.GetDim(2)});
    cur_executor->set_src_shape(&X_shape);
    cur_executor->set_dst_shape(&Y_shape);

    TensorImpl* sum_src = nullptr;
    TensorShape sum_shape;
    if (cur_executor->conv_param()->fuse_flag & ppl::kernel::x86::conv_fuse_flag::SUM) {
        sum_src = ctx->GetInput<TensorImpl>(ctx->GetInputCount() - 1);
        PPLNN_X86_DEBUG_TRACE("Input [sum_src]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(sum_src);

        sum_shape = *sum_src->GetShape();
        sum_shape.Reshape({sum_shape.GetDim(0), sum_shape.GetDim(1), 1, sum_shape.GetDim(2)});
        cur_executor->set_sum_src_shape(&sum_shape);
    }

    ppl::common::RetCode rc;
    rc = cur_executor->prepare();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "Prepare failed: " << ppl::common::GetRetCodeStr(rc);
        return rc;
    }

#ifdef DUMP_CONV
    fprintf(stderr, CASE_STRING_FMT() "\n", cur_executor->conv_param()->group, X_shape.GetDim(0),
            cur_executor->conv_param()->channels, X_shape.GetDim(2), X_shape.GetDim(3),
            cur_executor->conv_param()->num_output, Y_shape.GetDim(2), Y_shape.GetDim(3),
            cur_executor->conv_param()->kernel_h, cur_executor->conv_param()->kernel_w,
            cur_executor->conv_param()->stride_h, cur_executor->conv_param()->stride_w,
            cur_executor->conv_param()->pad_h, cur_executor->conv_param()->pad_w,
            cur_executor->conv_param()->dilation_h - 1, cur_executor->conv_param()->dilation_w - 1, GetName().c_str());
#endif

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

    cur_executor->set_temp_buffer(tmp_buffer);
    cur_executor->set_src(X->GetBufferPtr<float>());
    cur_executor->set_dst(Y->GetBufferPtr<float>());
    if (sum_src) {
        cur_executor->set_sum_src(sum_src->GetBufferPtr<float>());
    }

    rc = cur_executor->execute();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "Execute failed: " << ppl::common::GetRetCodeStr(rc);
        return rc;
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::x86
