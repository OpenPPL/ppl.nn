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

#include "ppl/nn/engines/x86/kernels/onnx/topk_kernel.h"
#include "ppl/nn/common/logger.h"
#include "ppl/kernel/x86/fp32/topk.h"

namespace ppl { namespace nn { namespace x86 {

uint64_t TopKKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    uint32_t axis =
        param_->axis < 0 ? param_->axis + ctx.GetInput<TensorImpl>(0)->GetShape()->GetDimCount() : param_->axis;
    return ppl::kernel::x86::topk_ndarray_fp32_get_buffer_bytes(ctx.GetInput<TensorImpl>(0)->GetShape(), axis);
}

ppl::common::RetCode TopKKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(x, 0);
    PPLNN_X86_REQUIRED_INPUT(k, 1);
    PPLNN_X86_REQUIRED_OUTPUT(values, 0);
    PPLNN_X86_REQUIRED_OUTPUT(indices, 1);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [x]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_X86_DEBUG_TRACE("Input [k]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(k);

    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    PPLNN_X86_REALLOC_TENSOR_BUFFER(values);
    PPLNN_X86_DEBUG_TRACE("Output [values]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(values);
    PPLNN_X86_REALLOC_TENSOR_BUFFER(indices);
    PPLNN_X86_DEBUG_TRACE("Output [indices]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(indices);

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
    PPLNN_X86_DEBUG_TRACE("buffer: %p\n", tmp_buffer);

    const int64_t k_value = ctx->GetInput<TensorImpl>(1)->GetBufferPtr<const int64_t>()[0];
    uint32_t axis = param_->axis < 0 ? param_->axis + x->GetShape()->GetDimCount() : param_->axis;

    auto data_type = x->GetShape()->GetDataType();
    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        return kernel::x86::topk_ndarray_fp32(x->GetShape(), values->GetShape(), indices->GetShape(),
                                              x->GetBufferPtr<const float>(), k_value, axis, param_->largest,
                                              param_->sorted, tmp_buffer, values->GetBufferPtr<float>(),
                                              indices->GetBufferPtr<int64_t>());
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
