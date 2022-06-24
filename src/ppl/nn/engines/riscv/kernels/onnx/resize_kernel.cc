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

#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"

#include "ppl/nn/engines/riscv/kernels/onnx/resize_kernel.h"
#include "ppl/kernel/riscv/fp32/resize2d.h"
#include "ppl/kernel/riscv/fp16/resize2d.h"

namespace ppl { namespace nn { namespace riscv {

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
    PPLNN_RISCV_REQUIRED_INPUT(X, 0);
    PPLNN_RISCV_OPTIONAL_INPUT(roi, 1);
    PPLNN_RISCV_OPTIONAL_INPUT(scales, 2);
    PPLNN_RISCV_OPTIONAL_INPUT(sizes, 3);
    PPLNN_RISCV_REQUIRED_OUTPUT(Y, 0);

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [X]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(X);
    if (roi) {
        PPLNN_RISCV_DEBUG_TRACE("Input [roi]:\n");
        PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(roi);
    }
    if (scales) {
        PPLNN_RISCV_DEBUG_TRACE("Input [scales]:\n");
        PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(scales);
    }
    if (sizes) {
        PPLNN_RISCV_DEBUG_TRACE("Input [sizes]:\n");
        PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(sizes);
    }

    PPLNN_RISCV_DEBUG_TRACE("coord_trans_mode: %d\n", param_->coord_trans_mode);
    PPLNN_RISCV_DEBUG_TRACE("nearest_mode: %d\n", param_->nearest_mode);
    PPLNN_RISCV_DEBUG_TRACE("mode: %d\n", param_->mode);
    PPLNN_RISCV_DEBUG_TRACE("cubic_coeff_a: %f\n", param_->cubic_coeff_a);
    PPLNN_RISCV_DEBUG_TRACE("exclude_outside: %d\n", param_->exclude_outside);
    PPLNN_RISCV_DEBUG_TRACE("extrapolation_value: %f\n", param_->extrapolation_value);

    // PPLNN_RISCV_REALLOC_TENSOR_BUFFER(Y);
    PPLNN_RISCV_DEBUG_TRACE("Output [Y]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(Y);

    float scale_h = (float)Y->GetShape()->GetDim(2) / X->GetShape()->GetDim(2);
    float scale_w = (float)Y->GetShape()->GetDim(3) / X->GetShape()->GetDim(3);

    auto has_scales = scales && scales->GetShape()->GetDimCount() == 1 &&
        scales->GetShape()->GetDim(0) == X->GetShape()->GetDimCount();

    if (has_scales) {
        const float* scales_data = scales->GetBufferPtr<float>();
        scale_h = scales_data[2];
        scale_w = scales_data[3];
    }

    const auto data_type = X->GetShape()->GetDataType();
    const auto data_format = X->GetShape()->GetDataFormat();
    if (data_type != ppl::common::DATATYPE_FLOAT32 && data_type != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "only support fp32 && fp16 now.";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        if (param_->coord_trans_mode == param_->RESIZE_COORD_TRANS_MODE_PYTORCH_HALF_PIXEL &&
            param_->mode == param_->RESIZE_MODE_CUBIC) {
            if (data_type == ppl::common::DATATYPE_FLOAT16) {
                return kernel::riscv::resize2d_ndarray_pytorch_cubic_floor_fp16(
                    X->GetShape(), Y->GetShape(), X->GetBufferPtr<__fp16>(), scale_h, scale_w, param_->cubic_coeff_a,
                    Y->GetBufferPtr<__fp16>());
            } else if (data_type == ppl::common::DATATYPE_FLOAT32) {
                return kernel::riscv::resize2d_ndarray_pytorch_cubic_floor_fp32(
                    X->GetShape(), Y->GetShape(), X->GetBufferPtr<float>(), scale_h, scale_w, param_->cubic_coeff_a,
                    Y->GetBufferPtr<float>());
            }
        }
        if (param_->coord_trans_mode == param_->RESIZE_COORD_TRANS_MODE_PYTORCH_HALF_PIXEL &&
            param_->mode == param_->RESIZE_MODE_LINEAR) {
            if (data_type == ppl::common::DATATYPE_FLOAT16) {
                return kernel::riscv::resize2d_ndarray_pytorch_linear_floor_fp16(X->GetShape(), Y->GetShape(),
                                                                                 X->GetBufferPtr<__fp16>(), scale_h,
                                                                                 scale_w, Y->GetBufferPtr<__fp16>());
            } else if (data_type == ppl::common::DATATYPE_FLOAT32) {
                return kernel::riscv::resize2d_ndarray_pytorch_linear_floor_fp32(
                    X->GetShape(), Y->GetShape(), X->GetBufferPtr<float>(), scale_h, scale_w, Y->GetBufferPtr<float>());
            }
        }
        if (param_->coord_trans_mode == param_->RESIZE_COORD_TRANS_MODE_ASYMMETRIC &&
            param_->mode == param_->RESIZE_MODE_NEAREST) {
            if (data_type == ppl::common::DATATYPE_FLOAT16) {
                return kernel::riscv::resize2d_ndarray_asymmetric_nearest_floor_fp16(
                    X->GetShape(), Y->GetShape(), X->GetBufferPtr<__fp16>(), scale_h, scale_w,
                    Y->GetBufferPtr<__fp16>());
            } else if (data_type == ppl::common::DATATYPE_FLOAT32) {
                return kernel::riscv::resize2d_ndarray_asymmetric_nearest_floor_fp32(
                    X->GetShape(), Y->GetShape(), X->GetBufferPtr<float>(), scale_h, scale_w, Y->GetBufferPtr<float>());
            }
        }
    } else if (data_format == ppl::common::DATAFORMAT_N4CX && data_type == ppl::common::DATATYPE_FLOAT32) {
        if (param_->coord_trans_mode == param_->RESIZE_COORD_TRANS_MODE_PYTORCH_HALF_PIXEL &&
            param_->mode == param_->RESIZE_MODE_LINEAR) {
            return kernel::riscv::resize2d_nbcx_pytorch_linear_floor_fp32(
                X->GetShape(), Y->GetShape(), X->GetBufferPtr<float>(), scale_h, scale_w, Y->GetBufferPtr<float>());
        }
        if (param_->coord_trans_mode == param_->RESIZE_COORD_TRANS_MODE_ASYMMETRIC &&
            param_->mode == param_->RESIZE_MODE_NEAREST) {
            return kernel::riscv::resize2d_nbcx_asymmetric_nearest_floor_fp32(
                X->GetShape(), Y->GetShape(), X->GetBufferPtr<float>(), scale_h, scale_w, Y->GetBufferPtr<float>());
        }
    } else if (data_format == ppl::common::DATAFORMAT_N8CX && data_type == ppl::common::DATATYPE_FLOAT16) {
        if (param_->coord_trans_mode == param_->RESIZE_COORD_TRANS_MODE_PYTORCH_HALF_PIXEL &&
            param_->mode == param_->RESIZE_MODE_LINEAR) {
            return kernel::riscv::resize2d_nbcx_pytorch_linear_floor_fp16(
                X->GetShape(), Y->GetShape(), X->GetBufferPtr<__fp16>(), scale_h, scale_w, Y->GetBufferPtr<__fp16>());
        }
        if (param_->coord_trans_mode == param_->RESIZE_COORD_TRANS_MODE_ASYMMETRIC &&
            param_->mode == param_->RESIZE_MODE_NEAREST) {
            return kernel::riscv::resize2d_nbcx_asymmetric_nearest_floor_fp16(
                X->GetShape(), Y->GetShape(), X->GetBufferPtr<__fp16>(), scale_h, scale_w, Y->GetBufferPtr<__fp16>());
        }
    }

    LOG(ERROR) << "unsupported case.";
    return ppl::common::RC_UNSUPPORTED;
}

}}}; // namespace ppl::nn::riscv
