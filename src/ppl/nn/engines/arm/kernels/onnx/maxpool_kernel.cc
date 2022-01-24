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

#include "ppl/nn/engines/arm/kernels/onnx/maxpool_kernel.h"
#include "ppl/kernel/arm_server/maxpool/neon/maxpool.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode MaxPoolKernel::DoExecute(KernelExecContext* ctx) {
    auto indata = ctx->GetInput<TensorImpl>(0);
    auto outdata = ctx->GetOutput<TensorImpl>(0);

    const int32_t src_n = indata->GetShape()->GetDim(0);
    const int32_t src_c = indata->GetShape()->GetDim(1);
    const int32_t src_h = indata->GetShape()->GetDim(2);
    const int32_t src_w = indata->GetShape()->GetDim(3);
    const int32_t dst_h = outdata->GetShape()->GetDim(2);
    const int32_t dst_w = outdata->GetShape()->GetDim(3);

    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t pad_h;
    int32_t pad_w;
    int32_t dilation_h;
    int32_t dilation_w;
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

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [in]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(indata);
    PPLNN_ARM_DEBUG_TRACE("Input [out]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(outdata);
    PPLNN_ARM_DEBUG_TRACE("kernel_shape: %d %d\n", kernel_h, kernel_w);
    PPLNN_ARM_DEBUG_TRACE("dilations: %d %d\n", dilation_h, dilation_w);
    PPLNN_ARM_DEBUG_TRACE("strides: %d %d\n", stride_h, stride_w);
    PPLNN_ARM_DEBUG_TRACE("pads: %d %d\n", pad_h, pad_w);
    PPLNN_ARM_DEBUG_TRACE("ceil: %d\n", param_->ceil_mode);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_format = indata->GetShape()->GetDataFormat();
    const auto data_type = indata->GetShape()->GetDataType();

    if (data_format == ppl::common::DATAFORMAT_N4CX) {
        if (data_type == ppl::common::DATATYPE_FLOAT32) {
            if (MayUseISA(ppl::common::ISA_ARMV8)) {
                return ppl::kernel::arm_server::neon::maxpool2d_n4cx_fp32(
                    indata->GetBufferPtr<const float>(), outdata->GetBufferPtr<float>(), src_n, src_c, src_h, src_w,
                    dst_h, dst_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
                    param_->global_pooling);
            } else {
                LOG(ERROR) << "unsupported datatype: " << ppl::common::GetDataTypeStr(data_type) << "with isa "
                           << GetISA() << ".";
                return ppl::common::RC_UNSUPPORTED;
            }
        }
    }
#ifdef PPLNN_USE_ARMV8_2_FP16
    else if (data_format == ppl::common::DATAFORMAT_N8CX) {
        if (data_type == ppl::common::DATATYPE_FLOAT16) {
            if (MayUseISA(ppl::common::ISA_ARMV8_2)) {
                return ppl::kernel::arm_server::neon::maxpool2d_n8cx_fp16(
                    indata->GetBufferPtr<const __fp16>(), outdata->GetBufferPtr<__fp16>(), src_n, src_c, src_h, src_w,
                    dst_h, dst_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
                    param_->global_pooling);
            } else {
                LOG(ERROR) << "unsupported datatype: " << ppl::common::GetDataTypeStr(data_type) << "with isa "
                           << GetISA() << ".";
                return ppl::common::RC_UNSUPPORTED;
            }
        }
    }
#endif

    LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
    LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format) << ".";
    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::arm
