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

#include "ppl/nn/engines/arm/kernels/onnx/concat_kernel.h"
#include "ppl/kernel/arm_server/concat/neon/concat.h"
#include "ppl/kernel/arm_server/common/memory.h"

namespace ppl { namespace nn { namespace arm {

bool ConcatKernel::CanDoExecute(const KernelExecContext& ctx) const {
    bool all_empty = true;
    for (uint32_t i = 0; i < ctx.GetInputCount(); i++) {
        auto tensor = ctx.GetInput<TensorImpl>(i);
        if (!tensor) {
            return false;
        }
        if (tensor->GetShape()->CalcBytesIncludingPadding() != 0) {
            all_empty = false;
        }
    }
    return !all_empty;
}

ppl::common::RetCode ConcatKernel::DoExecute(KernelExecContext* ctx) {
    src_list_.resize(ctx->GetInputCount());
    src_shape_list_.resize(ctx->GetInputCount());

    auto concat_result = ctx->GetOutput<TensorImpl>(0);

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    for (uint32_t i = 0; i < ctx->GetInputCount(); ++i) {
        auto input = ctx->GetInput<TensorImpl>(i);
        PPLNN_ARM_DEBUG_TRACE("Input [inputs[%u]]:\n", i);
        PPL_ARM_TENSOR_PRINT_DEBUG_MSG(input);
        src_shape_list_[i] = input->GetShape();
        src_list_[i] = input->GetBufferPtr();
    }
    PPLNN_ARM_DEBUG_TRACE("Output [concat_result]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(concat_result);
    PPLNN_ARM_DEBUG_TRACE("axis: %d\n", param_->axis);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    if (ctx->GetInputCount() == 1) {
        return ppl::kernel::arm_server::memory_copy(src_list_[0], src_shape_list_[0]->CalcBytesIncludingPadding(),
                                                    concat_result->GetBufferPtr<void>());
    }

    const int64_t real_axis =
        param_->axis < 0 ? param_->axis + ctx->GetInput<TensorImpl>(0)->GetShape()->GetDimCount() : param_->axis;

    return ppl::kernel::arm_server::neon::concat((const ppl::nn::TensorShape**)src_shape_list_.data(),
                                                 (const void**)src_list_.data(), ctx->GetInputCount(), real_axis,
                                                 concat_result->GetBufferPtr<void>());
}

}}} // namespace ppl::nn::arm
