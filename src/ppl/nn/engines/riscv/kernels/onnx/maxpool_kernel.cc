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

#include "ppl/nn/engines/riscv/kernels/onnx/maxpool_kernel.h"
#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/runtime/tensor_impl.h"

#include "ppl/kernel/riscv/fp16/maxpool2d.h"
#include "ppl/kernel/riscv/fp32/maxpool2d.h"

namespace ppl { namespace nn { namespace riscv {

ppl::common::RetCode MaxPoolKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_RISCV_REQUIRED_INPUT(X, 0);
    PPLNN_RISCV_REQUIRED_OUTPUT(Y, 0);
    PPLNN_RISCV_OPTIONAL_OUTPUT(Indices, 1);

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [X]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(X);
    PPLNN_RISCV_DEBUG_TRACE("Output [Y]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(Y);
    if (Indices) {
        PPLNN_RISCV_DEBUG_TRACE("Output [Indices]:\n");
        PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(Indices);
    }

    if (X->GetShape()->GetDimCount() != 4) {
        LOG(ERROR) << "only support 4-D tensor now.";
        return ppl::common::RC_UNSUPPORTED;
    }

    const int32_t src_h = X->GetShape()->GetDim(2);
    const int32_t src_w = X->GetShape()->GetDim(3);

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

    PPLNN_RISCV_DEBUG_TRACE("kernel_shape: %d %d\n", kernel_h, kernel_w);
    PPLNN_RISCV_DEBUG_TRACE("dilations: %d %d\n", dilation_h, dilation_w);
    PPLNN_RISCV_DEBUG_TRACE("strides: %d %d\n", stride_h, stride_w);
    PPLNN_RISCV_DEBUG_TRACE("pads: %d %d\n", pad_h, pad_w);
    PPLNN_RISCV_DEBUG_TRACE("mode: %d\n", param_->mode);
    PPLNN_RISCV_DEBUG_TRACE("ceil_mode: %d\n", param_->ceil_mode);
    PPLNN_RISCV_DEBUG_TRACE("global_pooling: %d\n", param_->global_pooling);
    ;

    const auto data_type = X->GetShape()->GetDataType();
    const auto data_format = X->GetShape()->GetDataFormat();

    if (ctx->GetOutputCount() == 1) {
        if (data_format == ppl::common::DATAFORMAT_N8CX && data_type == ppl::common::DATATYPE_FLOAT16) {
            return ppl::kernel::riscv::maxpool2d_n8chw_1x16_fp16(
                X->GetShape(), Y->GetShape(), kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,

                X->GetBufferPtr<const __fp16>(), Y->GetBufferPtr<__fp16>());
        } else if (data_format == ppl::common::DATAFORMAT_N4CX && data_type == ppl::common::DATATYPE_FLOAT32) {
            return ppl::kernel::riscv::maxpool2d_n4cx_1x16_fp32(
                X->GetShape(), Y->GetShape(), kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,

                X->GetBufferPtr<const float>(), Y->GetBufferPtr<float>());
        }
    } else if (ctx->GetOutputCount() == 2) {
        LOG(ERROR) << "unsupported maxpool with indices.";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::riscv
