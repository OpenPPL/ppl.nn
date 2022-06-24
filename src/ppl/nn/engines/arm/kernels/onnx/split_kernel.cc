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

#include "ppl/nn/engines/arm/kernels/onnx/split_kernel.h"
#include "ppl/kernel/arm_server/split/neon/split.h"
#include "ppl/kernel/arm_server/common/memory.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode SplitKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);

    std::vector<void*> dst_list(ctx->GetOutputCount());
    std::vector<const TensorShape*> dst_shape_list(ctx->GetOutputCount());

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [input]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(input);
    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto output = ctx->GetOutput<TensorImpl>(i);
        PPLNN_ARM_DEBUG_TRACE("Output [outputs[%u]]:\n", i);
        PPL_ARM_TENSOR_PRINT_DEBUG_MSG(output);
        dst_list[i] = output->GetBufferPtr<void>();
        dst_shape_list[i] = output->GetShape();
    }
    PPLNN_ARM_DEBUG_TRACE("axis: %d\n", param_->axis);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    if (ctx->GetOutputCount() == 1) {
        return ppl::kernel::arm_server::memory_copy(input->GetBufferPtr<void>(),
                                                    input->GetShape()->CalcBytesIncludingPadding(), dst_list[0]);
    }

    const int64_t real_axis =
        param_->axis < 0 ? param_->axis + ctx->GetInput<TensorImpl>(0)->GetShape()->GetDimCount() : param_->axis;

    return ppl::kernel::arm_server::neon::split(input->GetShape(), dst_shape_list.data(), input->GetBufferPtr<void>(),
                                                real_axis, ctx->GetOutputCount(), (void**)dst_list.data());
}

}}} // namespace ppl::nn::arm
