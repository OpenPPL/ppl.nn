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

#include "ppl/nn/engines/arm/kernels/onnx/conv2d_kernel.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/common/sys.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/arm/utils/macros.h"
#include "ppl/common/destructor.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode Conv2dKernel::DoExecute(KernelExecContext* ctx) {
    TensorImpl* X = ctx->GetInput<TensorImpl>(0);

    TensorImpl* Y = ctx->GetOutput<TensorImpl>(0);

    auto cur_executor = executor_;

    cur_executor->set_src_shape(X->GetShape());
    cur_executor->set_src(X->GetBufferPtr<void>());

    cur_executor->set_dst_shape(Y->GetShape());
    cur_executor->set_dst(Y->GetBufferPtr<void>());

    TensorImpl* S = nullptr;
    if (cur_executor->conv_param()->fuse_flag & ppl::kernel::arm_server::neon::conv_fuse_flag::SUM) {
        S = ctx->GetInput<TensorImpl>(ctx->GetInputCount() - 1);
        cur_executor->set_sum(S->GetBufferPtr<void>());
    }

    ppl::common::RetCode rc;
    rc = cur_executor->prepare();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "Conv kernel prepare failed: " << ppl::common::GetRetCodeStr(rc);
        return rc;
    }

#ifdef DUMP_CONV
#define CASE_STRING_FMT() "g%ld_mb%d_ic%ldih%diw%d_oc%ldoh%dow%d_kh%ldkw%ldsh%ldsw%ldph%ldpw%lddh%lddw%ld_n%s"
    fprintf(stderr, CASE_STRING_FMT() "\n", cur_executor->conv_param()->group, X->GetShape()->GetDim(0),
            cur_executor->conv_param()->channels, X->GetShape()->GetDim(2), X->GetShape()->GetDim(3),
            cur_executor->conv_param()->num_output, Y->GetShape()->GetDim(2), Y->GetShape()->GetDim(3),
            cur_executor->conv_param()->kernel_h, cur_executor->conv_param()->kernel_w,
            cur_executor->conv_param()->stride_h, cur_executor->conv_param()->stride_w,
            cur_executor->conv_param()->pad_h, cur_executor->conv_param()->pad_w,
            cur_executor->conv_param()->dilation_h - 1, cur_executor->conv_param()->dilation_w - 1, GetName().c_str());
#endif

    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_size = CalcTmpBufferSize(*ctx);
    auto status = GetArmDevice()->AllocTmpBuffer(tmp_buffer_size, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_size << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    ppl::common::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
        GetArmDevice()->FreeTmpBuffer(&tmp_buffer_desc);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;

    cur_executor->set_temp_buffer(tmp_buffer);

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [X]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(X);
    if (S) {
        PPLNN_ARM_DEBUG_TRACE("Input [S]:\n");
        PPL_ARM_TENSOR_PRINT_DEBUG_MSG(S);
    }
    PPLNN_ARM_DEBUG_TRACE("Output [Y]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(Y);
    PPLNN_ARM_DEBUG_TRACE("kernel_shape: %ld %ld\n", cur_executor->conv_param()->kernel_h,
                          cur_executor->conv_param()->kernel_w);
    PPLNN_ARM_DEBUG_TRACE("dilations: %ld %ld\n", cur_executor->conv_param()->dilation_h,
                          cur_executor->conv_param()->dilation_w);
    PPLNN_ARM_DEBUG_TRACE("strides: %ld %ld\n", cur_executor->conv_param()->stride_h,
                          cur_executor->conv_param()->stride_w);
    PPLNN_ARM_DEBUG_TRACE("pads: %ld %ld\n", cur_executor->conv_param()->pad_h, cur_executor->conv_param()->pad_w);
    PPLNN_ARM_DEBUG_TRACE("group: %ld\n", cur_executor->conv_param()->group);
    PPLNN_ARM_DEBUG_TRACE("channels: %ld\n", cur_executor->conv_param()->channels);
    PPLNN_ARM_DEBUG_TRACE("num_output: %ld\n", cur_executor->conv_param()->num_output);
    PPLNN_ARM_DEBUG_TRACE("buffer: %p\n", tmp_buffer);
    PPLNN_ARM_DEBUG_TRACE("fuse flag: %u\n", cur_executor->conv_param()->fuse_flag);
    PPLNN_ARM_DEBUG_TRACE("algo: %d\n", param_->mgr->algo_info().algo_type);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = X->GetShape()->GetDataType();
    if (data_type == ppl::common::DATATYPE_FLOAT16 && !MayUseISA(ppl::common::ISA_ARMV8_2)) {
        LOG(ERROR) << "fp16 needs isa >= armv8.2.";
        return ppl::common::RC_UNSUPPORTED;
    }

    rc = cur_executor->execute();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "Execute failed: " << ppl::common::GetRetCodeStr(rc);
        return rc;
    }

    return ppl::common::RC_SUCCESS;
}

uint64_t Conv2dKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    return executor_->cal_temp_buffer_size();
}

}}} // namespace ppl::nn::arm
