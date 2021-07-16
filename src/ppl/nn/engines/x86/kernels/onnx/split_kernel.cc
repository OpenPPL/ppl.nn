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

#include "ppl/nn/engines/x86/kernels/onnx/split_kernel.h"
#include "ppl/nn/engines/x86/macros.h"

#include "ppl/kernel/x86/fp32/split.h"
#include "ppl/kernel/x86/int64/split.h"
#include "ppl/kernel/x86/bool/split.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode SplitKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);

    std::vector<void*> dst_list(ctx->GetOutputCount());
    std::vector<const TensorShape*> dst_shape_list(ctx->GetOutputCount());

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [input]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(input);
    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto output = ctx->GetOutput<TensorImpl>(i);
        PPLNN_X86_DEBUG_TRACE("Output [outputs[%u]]:\n", i);
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
        dst_list[i] = output->GetBufferPtr<void>();
        dst_shape_list[i] = &output->GetShape();
    }
    PPLNN_X86_DEBUG_TRACE("axis: %d\n", param_->axis);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const int32_t real_axis =
        param_->axis < 0 ? param_->axis + ctx->GetInput<TensorImpl>(0)->GetShape().GetDimCount() : param_->axis;

    auto data_type = input->GetShape().GetDataType();
    auto data_format = input->GetShape().GetDataFormat();
    if (ppl::common::GetSizeOfDataType(data_type) == 4 && data_format == ppl::common::DATAFORMAT_N16CX &&
        real_axis == 1 && MayUseISA(ppl::common::ISA_X86_AVX)) {
        bool interleave_channels = false;
        for (uint32_t i = 0; i < dst_shape_list.size() - 1; i++) {
            if (dst_shape_list[i]->GetDim(1) % 16 != 0) {
                interleave_channels = true;
                break;
            }
        }
        if (interleave_channels) {
            return kernel::x86::split_n16cx_interleave_channels_fp32_avx(
                &input->GetShape(), dst_shape_list.data(), input->GetBufferPtr<float>(), real_axis,
                ctx->GetOutputCount(), 1, (float**)dst_list.data());
        }
    }

    if (ppl::common::GetSizeOfDataType(data_type) == 4) {
        if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
            return kernel::x86::split_ndarray_fp32(&input->GetShape(), dst_shape_list.data(),
                                                   input->GetBufferPtr<float>(), param_->axis, ctx->GetOutputCount(),
                                                   (float**)dst_list.data());
        } else if (data_format == ppl::common::DATAFORMAT_N16CX) {
            return kernel::x86::split_n16cx_fp32(&input->GetShape(), dst_shape_list.data(),
                                                 input->GetBufferPtr<float>(), param_->axis, ctx->GetOutputCount(),
                                                 (float**)dst_list.data());
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format);
        }
    } else if (ppl::common::GetSizeOfDataType(data_type) == 8) {
        if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
            return kernel::x86::split_ndarray_int64(&input->GetShape(), dst_shape_list.data(),
                                                    input->GetBufferPtr<int64_t>(), param_->axis, ctx->GetOutputCount(),
                                                    (int64_t**)dst_list.data());
        } else if (data_format == ppl::common::DATAFORMAT_N16CX) {
            return kernel::x86::split_n16cx_int64(&input->GetShape(), dst_shape_list.data(),
                                                  input->GetBufferPtr<int64_t>(), param_->axis, ctx->GetOutputCount(),
                                                  (int64_t**)dst_list.data());
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format);
        }
    } else if (ppl::common::GetSizeOfDataType(data_type) == 1) {
        if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
            return kernel::x86::split_ndarray_bool(&input->GetShape(), dst_shape_list.data(),
                                                   input->GetBufferPtr<uint8_t>(), param_->axis, ctx->GetOutputCount(),
                                                   (uint8_t**)dst_list.data());
        } else if (data_format == ppl::common::DATAFORMAT_N16CX) {
            return kernel::x86::split_n16cx_bool(&input->GetShape(), dst_shape_list.data(),
                                                 input->GetBufferPtr<uint8_t>(), param_->axis, ctx->GetOutputCount(),
                                                 (uint8_t**)dst_list.data());
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format);
        }
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
