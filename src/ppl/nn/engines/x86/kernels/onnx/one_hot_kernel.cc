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

#include "ppl/nn/engines/x86/kernels/onnx/one_hot_kernel.h"
#include "ppl/kernel/x86/int64/one_hot.h"
#include "ppl/kernel/x86/fp32/one_hot.h"
namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode OneHotKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(indices_tensor, 0);
    PPLNN_X86_REQUIRED_INPUT(depth_tensor, 1);
    PPLNN_X86_REQUIRED_INPUT(values_tensor, 2);
    PPLNN_X86_REQUIRED_OUTPUT(y, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [indices]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(indices_tensor);
    PPLNN_X86_DEBUG_TRACE("Input [depth]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(depth_tensor);
    PPLNN_X86_DEBUG_TRACE("Input [values]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(values_tensor);

    PPLNN_X86_DEBUG_TRACE("axis: %d\n", param_->axis);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());
    PPLNN_X86_REALLOC_TENSOR_BUFFER(y);
    PPLNN_X86_DEBUG_TRACE("Output [y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(y);

    const auto* indices_shape = indices_tensor->GetShape();
    const auto* depth_shape = depth_tensor->GetShape();
    const auto* values_shape = values_tensor->GetShape();
    const auto data_type = ctx->GetInput<TensorImpl>(2)->GetShape()->GetDataType(); // decide on values_tensor type
    const auto data_format = ctx->GetInput<TensorImpl>(0)->GetShape()->GetDataFormat(); // only support ndarray
    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        if (data_type == ppl::common::DATATYPE_INT64) {
            return ppl::kernel::x86::one_hot_ndarray_int64(
                indices_shape, depth_shape, values_shape, indices_tensor->GetBufferPtr(), depth_tensor->GetBufferPtr(),
                values_tensor->GetBufferPtr<int64_t>(), param_->axis, y->GetBufferPtr<int64_t>());
        } else if (data_type == ppl::common::DATATYPE_FLOAT32) {
            return ppl::kernel::x86::one_hot_ndarray_fp32(
                indices_shape, depth_shape, values_shape, indices_tensor->GetBufferPtr(), depth_tensor->GetBufferPtr(),
                values_tensor->GetBufferPtr<float>(), param_->axis, y->GetBufferPtr<float>());
        } else {
            LOG(ERROR) << "unsupported data type " << ppl::common::GetDataTypeStr(data_type) << ".";
        }
    } else {
        LOG(ERROR) << "unsupported data format " << ppl::common::GetDataFormatStr(data_format) << ".";
    }
    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
