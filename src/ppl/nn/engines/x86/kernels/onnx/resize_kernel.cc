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
    auto& X = ctx.GetInput<TensorImpl>(0)->GetShape();
    if (X.GetBytesIncludingPadding() == 0) {
        return false;
    }

    auto& scales = ctx.GetInput<TensorImpl>(2)->GetShape();
    if (ctx.GetInputCount() == 3 && scales.GetBytesIncludingPadding() == 0) {
        return false;
    }

    if (ctx.GetInputCount() >= 4) {
        auto& sizes = ctx.GetInput<TensorImpl>(3)->GetShape();
        if (scales.GetBytesIncludingPadding() == 0 && sizes.GetBytesIncludingPadding() == 0) {
            return false;
        }
    }
    return true;
}

ppl::common::RetCode ResizeKernel::DoExecute(KernelExecContext* ctx) {
    auto X = ctx->GetInput<TensorImpl>(0);
    auto Y = ctx->GetOutput<TensorImpl>(0);

    float scale_h = (float)Y->GetShape().GetDim(2) / X->GetShape().GetDim(2);
    float scale_w = (float)Y->GetShape().GetDim(3) / X->GetShape().GetDim(3);
    auto scales = ctx->GetInput<TensorImpl>(2);
    if (!scales->GetShape().IsEmpty()) {
        const float* scales_data = scales->GetBufferPtr<float>();
        scale_h = scales_data[2];
        scale_w = scales_data[3];
    }

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [X]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(X);
    auto roi = ctx->GetInput<TensorImpl>(1);
    PPLNN_X86_DEBUG_TRACE("Input [roi]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(roi);
    PPLNN_X86_DEBUG_TRACE("Input [scales]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(scales);
    if (ctx->GetInputCount() == 4) {
        auto sizes = ctx->GetInput<TensorImpl>(3);
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
            if (MayUseISA(ppl::common::ISA_X86_AVX512)) {
                return kernel::x86::resize2d_n16cx_pytorch_2linear_floor_fp32_avx512(&X->GetShape(), &Y->GetShape(),
                                                                                     X->GetBufferPtr<float>(), scale_h,
                                                                                     scale_w, Y->GetBufferPtr<float>());
            }
            if (MayUseISA(ppl::common::ISA_X86_AVX)) {
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
            if (MayUseISA(ppl::common::ISA_X86_AVX512)) {
                return kernel::x86::reisze2d_n16cx_asymmetric_nearest_floor_fp32_avx512(
                    &X->GetShape(), &Y->GetShape(), X->GetBufferPtr<float>(), scale_h, scale_w,
                    Y->GetBufferPtr<float>());
            } else if (MayUseISA(ppl::common::ISA_X86_AVX)) {
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
