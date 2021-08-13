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

#include "ppl/nn/engines/x86/kernels/onnx/resize_kernel.h"
#include "ppl/kernel/x86/fp32/resize2d.h"

namespace ppl { namespace nn { namespace x86 {

bool ResizeKernel::CanDoExecute(const KernelExecContext& ctx) const {
    auto& X_shape = ctx.GetInput<TensorImpl>(0)->GetShape();
    if (X_shape.GetBytesIncludingPadding() == 0) {
        return false;
    }

    auto scales = ctx.GetInputCount() > 2 ? ctx.GetInput<TensorImpl>(2) : nullptr;
    auto sizes = ctx.GetInputCount() > 3 ? ctx.GetInput<TensorImpl>(3) : nullptr;

    auto has_size = sizes && sizes->GetShape().GetDimCount() == 1 && sizes->GetShape().GetDim(0) == X_shape.GetDimCount();
    auto has_scales = scales && scales->GetShape().GetDimCount() == 1 && scales->GetShape().GetDim(0) == X_shape.GetDimCount();

    if (has_scales && has_size) {
        return false;
    }

    if (!has_scales && !has_size) {
        return false;
    }

    return true;
}

ppl::common::RetCode ResizeKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(X, 0);
    PPLNN_X86_OPTIONAL_INPUT(roi, 1);
    PPLNN_X86_OPTIONAL_INPUT(scales, 2);
    PPLNN_X86_OPTIONAL_INPUT(sizes, 3);
    PPLNN_X86_REQUIRED_OUTPUT(Y, 0);

    float scale_h = (float)Y->GetShape().GetDim(2) / X->GetShape().GetDim(2);
    float scale_w = (float)Y->GetShape().GetDim(3) / X->GetShape().GetDim(3);

    auto has_scales = scales && scales->GetShape().GetDimCount() == 1 && scales->GetShape().GetDim(0) == X->GetShape().GetDimCount();

    if (has_scales) {
        const float* scales_data = scales->GetBufferPtr<float>();
        scale_h = scales_data[2];
        scale_w = scales_data[3];
    }
    

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [X]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(X);
    if (roi) {
        PPLNN_X86_DEBUG_TRACE("Input [roi]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(roi);
    }
    if (scales) {
        PPLNN_X86_DEBUG_TRACE("Input [scales]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(scales);
    }
    if (sizes) {
        PPLNN_X86_DEBUG_TRACE("Input [sizes]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(sizes);
    }
    PPLNN_X86_DEBUG_TRACE("Output [Y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);
    PPLNN_X86_DEBUG_TRACE("coord_trans_mode: %d\n", param_->coord_trans_mode);
    PPLNN_X86_DEBUG_TRACE("nearest_mode: %d\n", param_->nearest_mode);
    PPLNN_X86_DEBUG_TRACE("mode: %d\n", param_->mode);
    PPLNN_X86_DEBUG_TRACE("cubic_coeff_a: %f\n", param_->cubic_coeff_a);
    PPLNN_X86_DEBUG_TRACE("exclude_outside: %d\n", param_->exclude_outside);
    PPLNN_X86_DEBUG_TRACE("extrapolation_value: %f\n", param_->extrapolation_value);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = X->GetShape().GetDataType();
    if (data_type != ppl::common::DATATYPE_FLOAT32) {
        LOG(ERROR) << "only support fp32 now.";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (param_->coord_trans_mode == param_->RESIZE_COORD_TRANS_MODE_PYTORCH_HALF_PIXEL &&
        param_->mode == param_->RESIZE_MODE_CUBIC && X->GetShape().GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        return kernel::x86::reisze2d_ndarray_pytorch_cubic_floor_fp32(&X->GetShape(), &Y->GetShape(),
                                                                      X->GetBufferPtr<float>(), scale_h, scale_w,
                                                                      param_->cubic_coeff_a, Y->GetBufferPtr<float>());
    }
    if (param_->coord_trans_mode == param_->RESIZE_COORD_TRANS_MODE_PYTORCH_HALF_PIXEL &&
        param_->mode == param_->RESIZE_MODE_LINEAR) {
        if (X->GetShape().GetDataFormat() == ppl::common::DATAFORMAT_N16CX) {
            if (false) {
            }
#ifdef PPL_USE_X86_AVX512
            else if (MayUseISA(ppl::common::ISA_X86_AVX512)) {
                return kernel::x86::resize2d_n16cx_pytorch_2linear_floor_fp32_avx512(&X->GetShape(), &Y->GetShape(),
                                                                                     X->GetBufferPtr<float>(), scale_h,
                                                                                     scale_w, Y->GetBufferPtr<float>());
            }
#endif
            else if (MayUseISA(ppl::common::ISA_X86_AVX)) {
                return kernel::x86::resize2d_n16chw_pytorch_2linear_floor_fp32_avx(&X->GetShape(), &Y->GetShape(),
                                                                                   X->GetBufferPtr<float>(), scale_h,
                                                                                   scale_w, Y->GetBufferPtr<float>());
            }
        } else if (X->GetShape().GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
            return kernel::x86::reisze2d_ndarray_pytorch_linear_floor_fp32(
                &X->GetShape(), &Y->GetShape(), X->GetBufferPtr<float>(), scale_h, scale_w, Y->GetBufferPtr<float>());
        }
    }
    if (param_->coord_trans_mode == param_->RESIZE_COORD_TRANS_MODE_ASYMMETRIC &&
        param_->mode == param_->RESIZE_MODE_NEAREST) {
        if (X->GetShape().GetDataFormat() == ppl::common::DATAFORMAT_N16CX) {
            if (false) {
            }
#ifdef PPL_USE_X86_AVX512
            else if (MayUseISA(ppl::common::ISA_X86_AVX512)) {
                return kernel::x86::reisze2d_n16cx_asymmetric_nearest_floor_fp32_avx512(
                    &X->GetShape(), &Y->GetShape(), X->GetBufferPtr<float>(), scale_h, scale_w,
                    Y->GetBufferPtr<float>());
            }
#endif
            else if (MayUseISA(ppl::common::ISA_X86_AVX)) {
                return kernel::x86::reisze2d_n16cx_asymmetric_nearest_floor_fp32_avx(&X->GetShape(), &Y->GetShape(),
                                                                                     X->GetBufferPtr<float>(), scale_h,
                                                                                     scale_w, Y->GetBufferPtr<float>());
            }
        } else if (X->GetShape().GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
            if ((X->GetShape().GetDim(2) * 2) == Y->GetShape().GetDim(2) &&
                (X->GetShape().GetDim(3) * 2) == Y->GetShape().GetDim(3)) {
                return kernel::x86::reisze2d_ndarray_asymmetric_nearest_floor_2times_fp32_sse(
                    &X->GetShape(), &Y->GetShape(), X->GetBufferPtr<float>(), scale_h, scale_w,
                    Y->GetBufferPtr<float>());
            } else {
                return kernel::x86::reisze2d_ndarray_asymmetric_nearest_floor_fp32(&X->GetShape(), &Y->GetShape(),
                                                                                   X->GetBufferPtr<float>(), scale_h,
                                                                                   scale_w, Y->GetBufferPtr<float>());
            }
        }
    }
    LOG(ERROR) << "unsupported case";
    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
