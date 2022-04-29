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
#include "ppl/nn/utils/destructor.h"
#include "ppl/nn/engines/x86/kernels/pmx/post_depthwise_conv2d_kernel.h"

#define CASE_STRING_FMT() \
    "g%" PRId64 \
    "_mb%" PRId64 \
    "_ic%" PRId64 "ih%" PRId64 "iw%" PRId64 \
    "_oc%" PRId64 "oh%" PRId64 "ow%" PRId64 \
    "_kh%" PRId64 "kw%" PRId64 "sh%" PRId64 "sw%" PRId64 "ph%" PRId64 "pw%" PRId64 "dh%" PRId64 "dw%" PRId64 \
    "_kh%" PRId64 "kw%" PRId64 "sh%" PRId64 "sw%" PRId64 "ph%" PRId64 "pw%" PRId64 "dh%" PRId64 "dw%" PRId64 \
    "_n%s"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode PostDepthwiseConv2dKernel::SeparateExecute(KernelExecContext* ctx, TensorImpl* X, TensorImpl* Y) {
    auto& inter_shape = executor_->inter_shape();
    PPLNN_X86_DEBUG_TRACE("InterTensor:\n");
    PPLNN_X86_DEBUG_TRACE("DimCount: %u\n", inter_shape.GetDimCount());
    for (uint32_t i = 0; i < inter_shape.GetDimCount(); ++i) {
        PPLNN_X86_DEBUG_TRACE("\tdim[%u]: %ld\tpads: [%hu, %hu]\n", i, inter_shape.GetDim(i),
                               inter_shape.GetPadding0(i), inter_shape.GetPadding1(i));
    }
    PPLNN_X86_DEBUG_TRACE("DataType: %s\n", ppl::common::GetDataTypeStr(inter_shape.GetDataType()));
    PPLNN_X86_DEBUG_TRACE("DataFormat: %s\n", ppl::common::GetDataFormatStr(inter_shape.GetDataFormat()));

    BufferDesc inter_buffer_desc;
    auto status = GetX86Device()->Realloc(inter_shape, &inter_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc InterTensor size[" << inter_shape.GetBytesIncludingPadding() << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    utils::Destructor inter_buffer_guard([this, &inter_buffer_desc]() -> void {
            GetX86Device()->Free(&inter_buffer_desc);
    });
    auto inter_buffer = (float*)inter_buffer_desc.addr;

    {
        BufferDesc conv_tmp_buffer_desc;
        auto conv_tmp_buffer_size = executor_->conv2d_executor()->cal_temp_buffer_size();
        auto status = GetX86Device()->AllocTmpBuffer(conv_tmp_buffer_size, &conv_tmp_buffer_desc);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "alloc conv tmp buffer size[" << conv_tmp_buffer_size << "] for kernel[" << GetName()
                    << "] failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }
        utils::Destructor __tmp_buffer_guard([this, &conv_tmp_buffer_desc]() -> void {
            GetX86Device()->FreeTmpBuffer(&conv_tmp_buffer_desc);
        });
        auto conv_tmp_buffer = conv_tmp_buffer_desc.addr;
        PPLNN_X86_DEBUG_TRACE("conv buffer: %p\n", conv_tmp_buffer);

        executor_->conv2d_executor()->set_temp_buffer(conv_tmp_buffer);
        executor_->conv2d_executor()->set_src(X->GetBufferPtr<float>());
        executor_->conv2d_executor()->set_dst(inter_buffer);

        auto rc = executor_->conv2d_executor()->execute();
        if (ppl::common::RC_SUCCESS != rc) {
            LOG(ERROR) << "Execute failed: " << ppl::common::GetRetCodeStr(rc);
            return rc;
        }

         if (ctx->IsLastConsumerOfInput(0) && X->GetType() == TENSORTYPE_NORMAL) {
            PPLNN_X86_DEBUG_TRACE("Free Input [X]\n");
            X->FreeBuffer();
        }
    }

    {
        PPLNN_X86_REALLOC_TENSOR_BUFFER(Y);
        PPLNN_X86_DEBUG_TRACE("Output [Y]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);

        BufferDesc depthwise_conv_tmp_buffer_desc;
        auto depthwise_conv_tmp_buffer_size = executor_->depthwise_conv2d_executor()->cal_temp_buffer_size();
        auto status = GetX86Device()->AllocTmpBuffer(depthwise_conv_tmp_buffer_size, &depthwise_conv_tmp_buffer_desc);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "alloc depthwise conv tmp buffer size[" << depthwise_conv_tmp_buffer_size << "] for kernel[" << GetName()
                    << "] failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }
        utils::Destructor __tmp_buffer_guard([this, &depthwise_conv_tmp_buffer_desc]() -> void {
            GetX86Device()->FreeTmpBuffer(&depthwise_conv_tmp_buffer_desc);
        });
        auto depthwise_conv_tmp_buffer = depthwise_conv_tmp_buffer_desc.addr;
        PPLNN_X86_DEBUG_TRACE("depthwise conv buffer: %p\n", depthwise_conv_tmp_buffer);

        executor_->depthwise_conv2d_executor()->set_temp_buffer(depthwise_conv_tmp_buffer);
        executor_->depthwise_conv2d_executor()->set_src(inter_buffer);
        executor_->depthwise_conv2d_executor()->set_dst(Y->GetBufferPtr<float>());

        auto rc = executor_->depthwise_conv2d_executor()->execute();
        if (ppl::common::RC_SUCCESS != rc) {
            LOG(ERROR) << "Execute failed: " << ppl::common::GetRetCodeStr(rc);
            return rc;
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PostDepthwiseConv2dKernel::FuseExecute(KernelExecContext* ctx, TensorImpl* X, TensorImpl* Y) {
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
    utils::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
        GetX86Device()->FreeTmpBuffer(&tmp_buffer_desc);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;
    PPLNN_X86_DEBUG_TRACE("buffer: %p\n", tmp_buffer);

    executor_->set_temp_buffer(tmp_buffer);
    executor_->set_src(X->GetBufferPtr<float>());
    executor_->set_dst(Y->GetBufferPtr<float>());

    auto rc = executor_->execute();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "Execute failed: " << ppl::common::GetRetCodeStr(rc);
        return rc;
    }

    return ppl::common::RC_SUCCESS;
}

uint64_t PostDepthwiseConv2dKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    return executor_->cal_temp_buffer_size();
}

ppl::common::RetCode PostDepthwiseConv2dKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(X, 0);
    PPLNN_X86_REQUIRED_OUTPUT(Y, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());

    PPLNN_X86_DEBUG_TRACE("Input [X]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(X);

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
    PPLNN_X86_DEBUG_TRACE("fuse_flag: %ld, %ld\n",
        executor_->conv2d_executor()->conv_param()->fuse_flag,
        executor_->depthwise_conv2d_executor()->conv_param()->fuse_flag);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    executor_->set_src_shape(X->GetShape());
    executor_->set_dst_shape(Y->GetShape());

    ppl::common::RetCode rc;
    rc = executor_->prepare();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "Prepare failed: " << ppl::common::GetRetCodeStr(rc);
        return rc;
    }

#ifdef DUMP_CONV
    fprintf(stderr, CASE_STRING_FMT() "\n", executor_->conv2d_executor()->conv_param()->group, X->GetShape()->GetDim(0),
            executor_->conv2d_executor()->conv_param()->channels, X->GetShape()->GetDim(2), X->GetShape()->GetDim(3),
            executor_->conv2d_executor()->conv_param()->num_output, Y->GetShape()->GetDim(2), Y->GetShape()->GetDim(3),
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

    PPLNN_X86_DEBUG_TRACE("mode: %u\n", executor_->mode());
    if (executor_->mode() == ppl::kernel::x86::pd_conv2d_fp32_mode::SEPARATE) {
        return SeparateExecute(ctx, X, Y);
    } else {
        return FuseExecute(ctx, X, Y);
    }
}

}}} // namespace ppl::nn::x86
