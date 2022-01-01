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

#include "ppl/nn/engines/riscv/kernels/onnx/fc_kernel.h"
#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/runtime/tensor_impl.h"

namespace ppl { namespace nn { namespace riscv {

uint64_t FCKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    return executor_->cal_temp_buffer_size();
}

ppl::common::RetCode FCKernel::DoExecute(KernelExecContext* ctx) {
    TensorImpl* A = ctx->GetInput<TensorImpl>(0);
    TensorImpl* Y = ctx->GetOutput<TensorImpl>(0);

    executor_->set_src_tensor(*A);
    executor_->set_dst_tensor(*Y);

    ppl::common::RetCode rc;
    rc = executor_->prepare();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "Prepare failed: " << ppl::common::GetRetCodeStr(rc);
        return rc;
    }

    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_size = CalcTmpBufferSize(*ctx);
    rc = GetRiscvDevice()->AllocTmpBuffer(tmp_buffer_size, &tmp_buffer_desc);
    if (rc != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_size << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(rc);
        return rc;
    }
    BufferDescGuard __tmp_buffer_guard(&tmp_buffer_desc, [this](BufferDesc* buffer) -> void {
        GetRiscvDevice()->FreeTmpBuffer(buffer);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;

    executor_->set_temp_buffer(tmp_buffer);

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [A]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(A);
    PPLNN_RISCV_DEBUG_TRACE("Output [Y]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(Y);
    PPLNN_RISCV_DEBUG_TRACE("channels: %ld\n", executor_->fc_param()->channels);
    PPLNN_RISCV_DEBUG_TRACE("num_output: %ld\n", executor_->fc_param()->num_output);
    PPLNN_RISCV_DEBUG_TRACE("buffer: %p\n", tmp_buffer);

    rc = executor_->execute();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "Execute failed: " << ppl::common::GetRetCodeStr(rc);
        return rc;
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::riscv