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

#include "ppl/nn/engines/x86/kernels/onnx/slice_kernel.h"

#include "ppl/kernel/x86/fp32/slice.h"
#include "ppl/kernel/x86/int64/slice.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode SliceKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(data, 0);
    PPLNN_X86_OPTIONAL_INPUT(starts_tensor, 1);
    PPLNN_X86_OPTIONAL_INPUT(ends_tensor, 2);
    PPLNN_X86_OPTIONAL_INPUT(axes_tensor, 3);
    PPLNN_X86_OPTIONAL_INPUT(steps_tensor, 4);
    PPLNN_X86_REQUIRED_OUTPUT(output, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [data]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(data);
    if (starts_tensor) {
        PPLNN_X86_DEBUG_TRACE("Input [starts]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(starts_tensor);
    }
    if (ends_tensor) {
        PPLNN_X86_DEBUG_TRACE("Input [ends]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(ends_tensor);
    }
    if (axes_tensor) {
        PPLNN_X86_DEBUG_TRACE("Input [axes]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(axes_tensor);
    }
    if (steps_tensor) {
        PPLNN_X86_DEBUG_TRACE("Input [steps]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(steps_tensor);
    }

    const int64_t axes_num = param_->starts.size();

    PPLNN_X86_DEBUG_TRACE("axes_num: %ld\n", axes_num);
    PPLNN_X86_DEBUG_TRACE("starts: ");
    for (int64_t i = 0; i < axes_num; ++i) {
        PPLNN_X86_DEBUG_TRACE("%ld ", param_->starts[i]);
    }
    PPLNN_X86_DEBUG_TRACE("\n");
    PPLNN_X86_DEBUG_TRACE("ends: ");
    for (int64_t i = 0; i < axes_num; ++i) {
        PPLNN_X86_DEBUG_TRACE("%ld ", param_->ends[i]);
    }
    PPLNN_X86_DEBUG_TRACE("\n");
    PPLNN_X86_DEBUG_TRACE("axes: ");
    for (int64_t i = 0; i < axes_num; ++i) {
        PPLNN_X86_DEBUG_TRACE("%ld ", param_->axes[i]);
    }
    PPLNN_X86_DEBUG_TRACE("\n");
    PPLNN_X86_DEBUG_TRACE("steps: ");
    for (int64_t i = 0; i < axes_num; ++i) {
        PPLNN_X86_DEBUG_TRACE("%ld ", param_->steps[i]);
    }
    PPLNN_X86_DEBUG_TRACE("\n");


    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    PPLNN_X86_REALLOC_TENSOR_BUFFER(output);
    PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);

    const ppl::common::datatype_t data_type = data->GetShape()->GetDataType();
    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        return kernel::x86::slice_ndarray_fp32(
            data->GetShape(), output->GetShape(), data->GetBufferPtr<float>(),
            param_->starts.data(), param_->steps.data(), param_->axes.data(),
            axes_num, output->GetBufferPtr<float>());
    } else if (data_type == ppl::common::DATATYPE_INT64) {
        return kernel::x86::slice_ndarray_int64(
            data->GetShape(), output->GetShape(), data->GetBufferPtr<int64_t>(),
            param_->starts.data(), param_->steps.data(), param_->axes.data(),
            axes_num, output->GetBufferPtr<int64_t>());
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
