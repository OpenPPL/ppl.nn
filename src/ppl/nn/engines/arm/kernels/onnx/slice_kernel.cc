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

#include "ppl/nn/engines/arm/kernels/onnx/slice_kernel.h"
#include "ppl/kernel/arm_server/slice/neon/slice.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode SliceKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_ARM_REQUIRED_INPUT(data, 0);
    PPLNN_ARM_REQUIRED_INPUT(starts_tensor, 1);
    PPLNN_ARM_REQUIRED_INPUT(ends_tensor, 2);
    PPLNN_ARM_OPTIONAL_INPUT(axes_tensor, 3);
    PPLNN_ARM_OPTIONAL_INPUT(steps_tensor, 4);
    PPLNN_ARM_REQUIRED_OUTPUT(output, 0);

    const int axes_num = ctx->GetInput<TensorImpl>(1)->GetShape()->GetDim(0);

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [data]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(data);
    PPLNN_ARM_DEBUG_TRACE("Input [starts]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(starts_tensor);
    PPLNN_ARM_DEBUG_TRACE("Input [ends]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(ends_tensor);
    if (axes_tensor) {
        PPLNN_ARM_DEBUG_TRACE("Input [axes]:\n");
        PPL_ARM_TENSOR_PRINT_DEBUG_MSG(axes_tensor);
    }
    if (steps_tensor) {
        PPLNN_ARM_DEBUG_TRACE("Input [steps]:\n");
        PPL_ARM_TENSOR_PRINT_DEBUG_MSG(steps_tensor);
    }
    PPLNN_ARM_DEBUG_TRACE("Output [output]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    // prepare starts, axes, steps
    auto starts = starts_tensor->GetBufferPtr<int64_t>();

    const int64_t* axes = nullptr;
    std::vector<int64_t> axes_vec;
    if (axes_tensor) {
        axes = axes_tensor->GetBufferPtr<int64_t>();
    } else {
        axes_vec.resize(axes_num);
        for (int i = 0; i < axes_num; i++) {
            axes_vec[i] = i;
        }
        axes = axes_vec.data();
    }

    std::vector<int64_t> steps_vec;
    const int64_t* steps = nullptr;
    if (steps_tensor) {
        steps = steps_tensor->GetBufferPtr<int64_t>();
    } else {
        steps_vec.resize(axes_num, 1);
        steps = steps_vec.data();
    }

    return ppl::kernel::arm_server::neon::slice(data->GetShape(), output->GetShape(), data->GetBufferPtr<void>(),
                                                starts, steps, axes, axes_num, output->GetBufferPtr<void>());
}

}}} // namespace ppl::nn::arm
