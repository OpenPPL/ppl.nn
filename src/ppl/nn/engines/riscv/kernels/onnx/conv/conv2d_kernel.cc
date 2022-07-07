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

#include "ppl/nn/engines/riscv/impls/include/ppl/kernel/riscv/common/conv2d.h"
#include "ppl/nn/engines/riscv/kernels/onnx/conv/conv2d_kernel.h"
#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/common/destructor.h"
#include "ppl/nn/common/logger.h"

#define CASE_STRING_FMT() "g%ld_mb%d_ic%ldih%diw%d_oc%ldoh%dow%d_kh%ldkw%ldsh%ldsw%ldph%ldpw%lddh%lddw%ld_n%s"

namespace ppl { namespace nn { namespace riscv {

uint64_t Conv2dKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    return executor_->cal_temp_buffer_size();
}

ppl::common::RetCode Conv2dKernel::DoExecute(KernelExecContext* ctx) {
    TensorImpl* X = ctx->GetInput<TensorImpl>(0);
    TensorImpl* Y = ctx->GetOutput<TensorImpl>(0);

    auto cur_executor = executor_;
    cur_executor->set_src_tensor(*X);
    cur_executor->set_dst_tensor(*Y);

    ppl::common::RetCode rc;
    rc = cur_executor->prepare();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "Prepare failed: " << ppl::common::GetRetCodeStr(rc);
        return rc;
    }

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
    cur_executor->set_temp_buffer(tmp_buffer);

    {
        // PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
        // PPLNN_RISCV_DEBUG_TRACE("Input [X]:\n");
        // PPLNN_RISCV_DEBUG_TRACE("Output [Y]:\n");
        // PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(Y);
        // PPLNN_RISCV_DEBUG_TRACE("kernel_shape: %ld %ld\n", cur_executor->conv_param()->kernel_h,
        //                         cur_executor->conv_param()->kernel_w);
        // PPLNN_RISCV_DEBUG_TRACE("dilations: %ld %ld\n", cur_executor->conv_param()->dilation_h,
        //                         cur_executor->conv_param()->dilation_w);
        // PPLNN_RISCV_DEBUG_TRACE("strides: %ld %ld\n", cur_executor->conv_param()->stride_h,
        //                         cur_executor->conv_param()->stride_w);
        // PPLNN_RISCV_DEBUG_TRACE("pads: %ld %ld\n", cur_executor->conv_param()->pad_h,
        //                         cur_executor->conv_param()->pad_w);
        // PPLNN_RISCV_DEBUG_TRACE("group: %ld\n", cur_executor->conv_param()->group);
        // PPLNN_RISCV_DEBUG_TRACE("channels: %ld\n", cur_executor->conv_param()->channels);
        // PPLNN_RISCV_DEBUG_TRACE("num_output: %ld\n", cur_executor->conv_param()->num_output);
        // PPLNN_RISCV_DEBUG_TRACE("buffer: %p\n", tmp_buffer);
    }

    rc = cur_executor->execute();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "Execute failed: " << ppl::common::GetRetCodeStr(rc);
        return rc;
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::riscv
