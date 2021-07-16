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

#include "ppl/nn/engines/x86/kernels/onnx/concat_kernel.h"
#include "ppl/nn/engines/x86/macros.h"
#include "ppl/kernel/x86/fp32/concat.h"
#include "ppl/kernel/x86/int64/concat.h"
#include "ppl/kernel/x86/bool/concat.h"

namespace ppl { namespace nn { namespace x86 {

bool ConcatKernel::CanDoExecute(const KernelExecContext& ctx) const {
    bool all_empty = true;
    for (uint32_t i = 0; i < ctx.GetInputCount(); i++) {
        auto tensor = ctx.GetInput<TensorImpl>(i);
        if (!tensor) {
            return false;
        }
        if (tensor->GetShape().GetBytesIncludingPadding() != 0) {
            all_empty = false;
        }
    }
    return !all_empty;
}

uint64_t ConcatKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    return 0;
}

ppl::common::RetCode ConcatKernel::DoExecute(KernelExecContext* ctx) {
    src_list_.resize(ctx->GetInputCount());
    src_shape_list_.resize(ctx->GetInputCount());

    auto concat_result = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    for (uint32_t i = 0; i < ctx->GetInputCount(); ++i) {
        auto input = ctx->GetInput<TensorImpl>(i);
        PPLNN_X86_DEBUG_TRACE("Input [inputs[%u]]:\n", i);
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(input);
        src_shape_list_[i] = &input->GetShape();
        src_list_[i] = input->GetBufferPtr();
    }
    PPLNN_X86_DEBUG_TRACE("Output [concat_result]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(concat_result);
    PPLNN_X86_DEBUG_TRACE("axis: %d\n", param_->axis);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    auto data_type = concat_result->GetShape().GetDataType();
    auto data_format = concat_result->GetShape().GetDataFormat();
    const int32_t real_axis =
        param_->axis < 0 ? param_->axis + ctx->GetInput<TensorImpl>(0)->GetShape().GetDimCount() : param_->axis;

    if (ppl::common::GetSizeOfDataType(data_type) == 4 && data_format == ppl::common::DATAFORMAT_N16CX &&
        real_axis == 1 && MayUseISA(ppl::common::ISA_X86_AVX)) {
        bool interleave_channels = false;
        for (uint32_t i = 0; i < src_shape_list_.size() - 1; i++) {
            if (src_shape_list_[i]->GetDim(1) % 16 != 0) {
                interleave_channels = true;
                break;
            }
        }
        if (interleave_channels) {
            return kernel::x86::concat_n16cx_interleave_channels_fp32_avx(
                src_shape_list_.data(), (const float**)src_list_.data(), ctx->GetInputCount(), real_axis, 1,
                concat_result->GetBufferPtr<float>());
        }
    }

    if (ppl::common::GetSizeOfDataType(data_type) == 4) {
        if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
            return kernel::x86::concat_ndarray_fp32(src_shape_list_.data(), (const float**)src_list_.data(),
                                                    ctx->GetInputCount(), param_->axis,
                                                    concat_result->GetBufferPtr<float>());
        } else if (data_format == ppl::common::DATAFORMAT_N16CX) {
            return kernel::x86::concat_n16cx_fp32(src_shape_list_.data(), (const float**)src_list_.data(),
                                                  ctx->GetInputCount(), param_->axis,
                                                  concat_result->GetBufferPtr<float>());
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format);
        }
    } else if (ppl::common::GetSizeOfDataType(data_type) == 8) {
        if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
            return kernel::x86::concat_ndarray_int64(src_shape_list_.data(), (const int64_t**)src_list_.data(),
                                                     ctx->GetInputCount(), param_->axis,
                                                     concat_result->GetBufferPtr<int64_t>());
        } else if (data_format == ppl::common::DATAFORMAT_N16CX) {
            return kernel::x86::concat_n16cx_int64(src_shape_list_.data(), (const int64_t**)src_list_.data(),
                                                   ctx->GetInputCount(), param_->axis,
                                                   concat_result->GetBufferPtr<int64_t>());
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format);
        }
    } else if (ppl::common::GetSizeOfDataType(data_type) == 1) {
        if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
            return kernel::x86::concat_ndarray_bool(src_shape_list_.data(), (const uint8_t**)src_list_.data(),
                                                    ctx->GetInputCount(), param_->axis,
                                                    concat_result->GetBufferPtr<uint8_t>());
        } else if (data_format == ppl::common::DATAFORMAT_N16CX) {
            return kernel::x86::concat_n16cx_bool(src_shape_list_.data(), (const uint8_t**)src_list_.data(),
                                                  ctx->GetInputCount(), param_->axis,
                                                  concat_result->GetBufferPtr<uint8_t>());
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format);
        }
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type);
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
