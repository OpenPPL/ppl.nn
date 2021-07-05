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

#include "ppl/nn/engines/x86/kernels/onnx/non_zero_kernel.h"
#include "ppl/nn/common/logger.h"

#include "ppl/kernel/x86/fp32/non_zero.h"
#include "ppl/kernel/x86/bool/non_zero.h"

namespace ppl { namespace nn { namespace x86 {

uint64_t NonZeroKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    if (ctx.GetInput<TensorImpl>(0)->GetShape().GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        return kernel::x86::non_zero_ndarray_fp32_get_buffer_bytes(&ctx.GetInput<TensorImpl>(0)->GetShape());
    } else if (ctx.GetInput<TensorImpl>(0)->GetShape().GetDataType() == ppl::common::DATATYPE_BOOL) {
        return kernel::x86::non_zero_ndarray_bool_get_buffer_bytes(&ctx.GetInput<TensorImpl>(0)->GetShape());
    }
    return 0;
}

ppl::common::RetCode NonZeroKernel::DoExecute(KernelExecContext* ctx) {
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

    auto x = ctx->GetInput<TensorImpl>(0);
    auto y = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [x]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_X86_DEBUG_TRACE("Output [y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(y);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    int64_t real_output_num = 0;
    ppl::common::RetCode ret = ppl::common::RC_SUCCESS;
    if (x->GetShape().GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        ret = kernel::x86::non_zero_ndarray_fp32(&x->GetShape(), x->GetBufferPtr<float>(), (int64_t*)tmp_buffer,
                                                 &real_output_num, y->GetBufferPtr<int64_t>());
    } else if (x->GetShape().GetDataType() == ppl::common::DATATYPE_BOOL) {
        ret = kernel::x86::non_zero_ndarray_bool(&x->GetShape(), x->GetBufferPtr<uint8_t>(), (int64_t*)tmp_buffer,
                                                 &real_output_num, y->GetBufferPtr<int64_t>());
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }

    if (ret == ppl::common::RC_SUCCESS) {
        y->GetShape().Reshape({x->GetShape().GetDimCount(), real_output_num}); // TODO: this will cause dynamic shape
    }
    return ret;
}

}}} // namespace ppl::nn::x86
