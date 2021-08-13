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

#include "ppl/nn/engines/x86/kernels/onnx/pad_kernel.h"

#include "ppl/kernel/x86/fp32/pad.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode PadKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(x, 0);
    PPLNN_X86_OPTIONAL_INPUT(constant, 1);
    PPLNN_X86_REQUIRED_OUTPUT(y, 0);

    float constant_value = constant ? (constant->GetBufferPtr<float>())[0] : 0;

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [x]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_X86_DEBUG_TRACE("Output [y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(y);
    PPLNN_X86_DEBUG_TRACE("pad mode: %d\n", param_->mode);
    PPLNN_X86_DEBUG_TRACE("constant_value: %f\n", constant_value);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const int dim_count = x->GetShape().GetDimCount();
    auto pads_data = ctx->GetInput<TensorImpl>(1)->GetBufferPtr<int64_t>();
    auto start_pads = pads_data;
    auto end_pads = pads_data + dim_count;

    if (x->GetShape().GetElementsExcludingPadding() ==
        y->GetShape().GetElementsExcludingPadding()) { // no padding at all, just copy
        if (x->GetEdge()->CalcConsumerCount() == 1 && x->GetType() == TENSORTYPE_NORMAL) {
            y->TransferBufferFrom(x);
        } else {
            memcpy(y->GetBufferPtr(), x->GetBufferPtr(), x->GetShape().GetBytesIncludingPadding());
        }
        return ppl::common::RC_SUCCESS;
    }

    auto data_type = x->GetShape().GetDataType();
    auto data_format = x->GetShape().GetDataFormat();
    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
            if (param_->mode == ppl::nn::common::PadParam::PAD_MODE_CONSTANT) {
                return kernel::x86::pad_ndarray_constant_fp32(&x->GetShape(), &y->GetShape(), x->GetBufferPtr<float>(),
                                                              start_pads, end_pads, constant_value,
                                                              y->GetBufferPtr<float>());
            } else if (param_->mode == ppl::nn::common::PadParam::PAD_MODE_REFLECT) {
                return kernel::x86::pad_ndarray_reflect_fp32(&x->GetShape(), &y->GetShape(), x->GetBufferPtr<float>(),
                                                             start_pads, end_pads, y->GetBufferPtr<float>());
            } else if (param_->mode == ppl::nn::common::PadParam::PAD_MODE_EDGE) {
                return kernel::x86::pad_ndarray_edge_fp32(&x->GetShape(), &y->GetShape(), x->GetBufferPtr<float>(),
                                                          start_pads, end_pads, y->GetBufferPtr<float>());
            } else {
                LOG(ERROR) << "invalid pad mode " << param_->mode << ".";
                return ppl::common::RC_INVALID_VALUE;
            }
        } else if (data_format == ppl::common::DATAFORMAT_N16CX) {
            if (param_->mode == ppl::nn::common::PadParam::PAD_MODE_CONSTANT) {
                return kernel::x86::pad_n16cx_constant_fp32(&x->GetShape(), &y->GetShape(), x->GetBufferPtr<float>(),
                                                            start_pads, end_pads, constant_value,
                                                            y->GetBufferPtr<float>());
            } else if (param_->mode == ppl::nn::common::PadParam::PAD_MODE_REFLECT) {
                return kernel::x86::pad_n16cx_reflect_fp32(&x->GetShape(), &y->GetShape(), x->GetBufferPtr<float>(),
                                                           start_pads, end_pads, y->GetBufferPtr<float>());
            } else if (param_->mode == ppl::nn::common::PadParam::PAD_MODE_EDGE) {
                return kernel::x86::pad_n16cx_edge_fp32(&x->GetShape(), &y->GetShape(), x->GetBufferPtr<float>(),
                                                        start_pads, end_pads, y->GetBufferPtr<float>());
            } else {
                LOG(ERROR) << "invalid pad mode " << param_->mode << ".";
                return ppl::common::RC_INVALID_VALUE;
            }
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format) << ".";
        }
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
