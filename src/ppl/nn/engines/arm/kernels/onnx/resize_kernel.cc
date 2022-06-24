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

#include <vector>

#include "ppl/nn/engines/arm/kernels/onnx/resize_kernel.h"
#include "ppl/kernel/arm_server/resize2d/neon/resize2d.h"

namespace ppl { namespace nn { namespace arm {

bool ResizeKernel::CanDoExecute(const KernelExecContext& ctx) const {
    auto& X_shape = *ctx.GetInput<TensorImpl>(0)->GetShape();
    if (X_shape.CalcBytesIncludingPadding() == 0) {
        return false;
    }

    auto scales = ctx.GetInputCount() > 2 ? ctx.GetInput<TensorImpl>(2) : nullptr;
    auto sizes = ctx.GetInputCount() > 3 ? ctx.GetInput<TensorImpl>(3) : nullptr;

    auto has_size =
        sizes && sizes->GetShape()->GetDimCount() == 1 && sizes->GetShape()->GetDim(0) == X_shape.GetDimCount();
    auto has_scales =
        scales && scales->GetShape()->GetDimCount() == 1 && scales->GetShape()->GetDim(0) == X_shape.GetDimCount();

    if (has_scales && has_size) {
        return false;
    }

    if (!has_scales && !has_size) {
        return false;
    }

    return true;
}

ppl::common::RetCode ResizeKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_ARM_REQUIRED_INPUT(X, 0);
    PPLNN_ARM_OPTIONAL_INPUT(roi, 1);
    PPLNN_ARM_OPTIONAL_INPUT(scales, 2);
    PPLNN_ARM_OPTIONAL_INPUT(sizes, 3);
    PPLNN_ARM_REQUIRED_OUTPUT(Y, 0);

    float scale_h = (float)Y->GetShape()->GetDim(2) / X->GetShape()->GetDim(2);
    float scale_w = (float)Y->GetShape()->GetDim(3) / X->GetShape()->GetDim(3);

    auto has_scales =
        scales && scales->GetShape()->GetDimCount() == 1 && scales->GetShape()->GetDim(0) == X->GetShape()->GetDimCount();

    if (has_scales) {
        const float* scales_data = scales->GetBufferPtr<float>();
        scale_h = scales_data[2];
        scale_w = scales_data[3];
    }

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [X]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(X);
    if (roi) {
        PPLNN_ARM_DEBUG_TRACE("Input [roi]:\n");
        PPL_ARM_TENSOR_PRINT_DEBUG_MSG(roi);
    }
    if (scales) {
        PPLNN_ARM_DEBUG_TRACE("Input [scales]:\n");
        PPL_ARM_TENSOR_PRINT_DEBUG_MSG(scales);
    }
    if (sizes) {
        PPLNN_ARM_DEBUG_TRACE("Input [sizes]:\n");
        PPL_ARM_TENSOR_PRINT_DEBUG_MSG(sizes);
    }
    PPLNN_ARM_DEBUG_TRACE("Output [Y]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(Y);
    PPLNN_ARM_DEBUG_TRACE("coord_trans_mode: %d\n", param_->coord_trans_mode);
    PPLNN_ARM_DEBUG_TRACE("nearest_mode: %d\n", param_->nearest_mode);
    PPLNN_ARM_DEBUG_TRACE("mode: %d\n", param_->mode);
    PPLNN_ARM_DEBUG_TRACE("cubic_coeff_a: %f\n", param_->cubic_coeff_a);
    PPLNN_ARM_DEBUG_TRACE("exclude_outside: %d\n", param_->exclude_outside);
    PPLNN_ARM_DEBUG_TRACE("extrapolation_value: %f\n", param_->extrapolation_value);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = X->GetShape()->GetDataType();
    if (data_type == ppl::common::DATATYPE_FLOAT16 && !MayUseISA(ppl::common::ISA_ARMV8_2)) {
        LOG(ERROR) << "fp16 needs isa >= armv8.2.";
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t dim_count = X->GetShape()->GetDimCount();
    if (dim_count != 4) {
        LOG(ERROR) << "only support resize2d now.";
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::kernel::arm_server::neon::resize2d(X->GetShape(), Y->GetShape(), X->GetBufferPtr<void>(), scale_h,
                                                   scale_w, param_, Y->GetBufferPtr<void>());
}

}}} // namespace ppl::nn::arm
