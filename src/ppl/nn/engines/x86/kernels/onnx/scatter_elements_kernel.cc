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

#include "ppl/nn/engines/x86/kernels/onnx/scatter_elements_kernel.h"
#include "ppl/nn/common/logger.h"

#include "ppl/kernel/x86/fp32/scatter_elements.h"
#include "ppl/kernel/x86/int64/scatter_elements.h"
namespace ppl { namespace nn { namespace x86 {

bool ScatterElementsKernel::CanDoExecute(const KernelExecContext& ctx) const {
    return ctx.GetInput<TensorImpl>(0)->GetShape().GetBytesIncludingPadding() != 0;
}

ppl::common::RetCode ScatterElementsKernel::DoExecute(KernelExecContext* ctx) {
    auto x = ctx->GetInput<TensorImpl>(0);
    auto indices = ctx->GetInput<TensorImpl>(1);
    auto updates = ctx->GetInput<TensorImpl>(2);
    auto y = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [x]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_X86_DEBUG_TRACE("Input [indices]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(indices);
    PPLNN_X86_DEBUG_TRACE("Input [updates]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(updates);
    PPLNN_X86_DEBUG_TRACE("Output [y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(y);
    PPLNN_X86_DEBUG_TRACE("axis: %d\n", param_->axis);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = x->GetShape().GetDataType();
    const auto data_format = x->GetShape().GetDataFormat();

    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        if (data_type == ppl::common::DATATYPE_FLOAT32) {
            return kernel::x86::scatter_elements_ndarray_fp32(
                &x->GetShape(), &indices->GetShape(), x->GetBufferPtr<float>(), indices->GetBufferPtr<int64_t>(),
                updates->GetBufferPtr<float>(), param_->axis, y->GetBufferPtr<float>());
        } else if (data_type == ppl::common::DATATYPE_INT64) {
            return kernel::x86::scatter_elements_ndarray_int64(
                &x->GetShape(), &indices->GetShape(), x->GetBufferPtr<int64_t>(), indices->GetBufferPtr<int64_t>(),
                updates->GetBufferPtr<int64_t>(), param_->axis, y->GetBufferPtr<int64_t>());
        } else {
            LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
        }
    } else {
        LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format);
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
