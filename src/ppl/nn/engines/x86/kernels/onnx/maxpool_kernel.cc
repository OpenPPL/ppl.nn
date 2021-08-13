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

#include "ppl/nn/engines/x86/kernels/onnx/maxpool_kernel.h"
#include "ppl/kernel/x86/fp32/maxpool2d.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode MaxPoolKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(X, 0);
    PPLNN_X86_REQUIRED_OUTPUT(Y, 0);
    PPLNN_X86_OPTIONAL_OUTPUT(Indices, 1);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [X]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(X);
    PPLNN_X86_DEBUG_TRACE("Output [Y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);
    if (Indices) {
        PPLNN_X86_DEBUG_TRACE("Output [Indices]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(Indices);
    }

    if (X->GetShape().GetDimCount() != 4) {
        LOG(ERROR) << "only support 4-D tensor now.";
        return ppl::common::RC_UNSUPPORTED;
    }

    const int32_t src_h = X->GetShape().GetDim(2);
    const int32_t src_w = X->GetShape().GetDim(3);

    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t pad_h;
    int32_t pad_w;
    int32_t dilation_h;
    int32_t dilation_w;
    if (param_->global_pooling) {
        kernel_h = src_h;
        kernel_w = src_w;
        stride_h = src_h;
        stride_w = src_w;
        pad_h = 0;
        pad_w = 0;
        dilation_h = 1;
        dilation_w = 1;
    } else {
        kernel_h = param_->kernel_shape[0];
        kernel_w = param_->kernel_shape[1];
        stride_h = param_->strides.size() >= 1 ? param_->strides[0] : 1;
        stride_w = param_->strides.size() >= 2 ? param_->strides[1] : 1;
        pad_h = param_->pads.size() >= 1 ? param_->pads[0] : 0;
        pad_w = param_->pads.size() >= 2 ? param_->pads[1] : 0;
        if ((param_->pads.size() >= 3 && param_->pads[2] != pad_h) ||
            (param_->pads.size() >= 4 && param_->pads[3] != pad_w)) {
            LOG(ERROR) << "only support symmetrical pads now.";
            return ppl::common::RC_UNSUPPORTED;
        }
        dilation_h = param_->dilations.size() >= 1 ? param_->dilations[0] : 1;
        dilation_w = param_->dilations.size() >= 2 ? param_->dilations[1] : 1;
        if (dilation_h != 1 || dilation_w != 1) {
            LOG(ERROR) << "only support dilation = 1 now.";
            return ppl::common::RC_UNSUPPORTED;
        }
    }

    PPLNN_X86_DEBUG_TRACE("kernel_shape: %d %d\n", kernel_h, kernel_w);
    PPLNN_X86_DEBUG_TRACE("dilations: %d %d\n", dilation_h, dilation_w);
    PPLNN_X86_DEBUG_TRACE("strides: %d %d\n", stride_h, stride_w);
    PPLNN_X86_DEBUG_TRACE("pads: %d %d\n", pad_h, pad_w);
    PPLNN_X86_DEBUG_TRACE("mode: %d\n", param_->mode);
    PPLNN_X86_DEBUG_TRACE("ceil_mode: %d\n", param_->ceil_mode);
    PPLNN_X86_DEBUG_TRACE("global_pooling: %d\n", param_->global_pooling);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = X->GetShape().GetDataType();
    const auto data_format = X->GetShape().GetDataFormat();

    if (ctx->GetOutputCount() == 1) {
        if (data_format == ppl::common::DATAFORMAT_N16CX) {
            if (data_type == ppl::common::DATATYPE_FLOAT32) {
                if (false) {
                }
#ifdef PPL_USE_X86_AVX512
                else if (MayUseISA(ppl::common::ISA_X86_AVX512)) {
                    return ppl::kernel::x86::maxpool2d_n16chw_blk1x16_fp32_avx512(
                        &X->GetShape(), &Y->GetShape(), X->GetBufferPtr<float>(), kernel_h, kernel_w, stride_h,
                        stride_w, pad_h, pad_w, Y->GetBufferPtr<float>());
                }
#endif
                else if (MayUseISA(ppl::common::ISA_X86_AVX)) {
                    return ppl::kernel::x86::maxpool2d_n16chw_blk1x8_fp32_avx(
                        &X->GetShape(), &Y->GetShape(), X->GetBufferPtr<float>(), kernel_h, kernel_w, stride_h,
                        stride_w, pad_h, pad_w, Y->GetBufferPtr<float>());
                } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
                    return ppl::kernel::x86::maxpool2d_n16chw_blk1x4_fp32_sse(
                        &X->GetShape(), &Y->GetShape(), X->GetBufferPtr<float>(), kernel_h, kernel_w, stride_h,
                        stride_w, pad_h, pad_w, Y->GetBufferPtr<float>());
                } else {
                    LOG(ERROR) << "get unsupported isa " << GetISA() << ".";
                }
            } else {
                LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
            }
        } else if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
            if (data_type == ppl::common::DATATYPE_FLOAT32) {
                return ppl::kernel::x86::maxpool2d_nchw_normal_fp32(
                    &X->GetShape(), &Y->GetShape(), X->GetBufferPtr<float>(), kernel_h, kernel_w, stride_h, stride_w,
                    pad_h, pad_w, Y->GetBufferPtr<float>());
            } else {
                LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
            }
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format) << ".";
        }
    } else if (ctx->GetOutputCount() == 2) {
        if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
            if (data_type == ppl::common::DATATYPE_FLOAT32) {
                return ppl::kernel::x86::maxpool2d_nchw_with_indices_fp32(
                    &X->GetShape(), &Y->GetShape(), X->GetBufferPtr<float>(), kernel_h, kernel_w, stride_h, stride_w,
                    pad_h, pad_w, Y->GetBufferPtr<float>(), Indices->GetBufferPtr<int64_t>());
            } else {
                LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
            }
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format) << ".";
        }
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
