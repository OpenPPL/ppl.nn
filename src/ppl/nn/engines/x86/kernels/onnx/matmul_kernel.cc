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

#include <deque>

#include "ppl/nn/engines/x86/kernels/onnx/matmul_kernel.h"
#include "ppl/kernel/x86/fp32/matmul.h"

namespace ppl { namespace nn { namespace x86 {

uint64_t MatMulKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    const auto& A = ctx.GetInput<TensorImpl>(0)->GetShape();
    const auto& B = ctx.GetInput<TensorImpl>(1)->GetShape();

    return kernel::x86::matmul_ndarray_fp32_get_buffer_bytes(&A, &B, GetISA());
}

ppl::common::RetCode MatMulKernel::DoExecute(KernelExecContext* ctx) {
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

    auto A = ctx->GetInput<TensorImpl>(0);
    auto B = ctx->GetInput<TensorImpl>(1);
    auto Y = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [A]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(A);
    PPLNN_X86_DEBUG_TRACE("Input [B]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(B);
    PPLNN_X86_DEBUG_TRACE("Output [Y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = A->GetShape().GetDataType();
    const auto data_format = A->GetShape().GetDataFormat();

    if (data_type == ppl::common::DATATYPE_FLOAT32 && data_format == ppl::common::DATAFORMAT_NDARRAY) {
        return kernel::x86::matmul_ndarray_fp32(&A->GetShape(), &B->GetShape(), &Y->GetShape(),
                                                A->GetBufferPtr<float>(), B->GetBufferPtr<float>(), GetISA(),
                                                tmp_buffer, Y->GetBufferPtr<float>());
    } else {
        LOG(ERROR) << "only support fp32 ndarray now.";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
