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

#include "ppl/nn/engines/riscv/kernels/onnx/concat_kernel.h"
#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"

#include "ppl/kernel/riscv/fp16/concat.h"
#include "ppl/kernel/riscv/fp32/concat.h"
#include "ppl/kernel/riscv/int64/concat.h"

namespace ppl { namespace nn { namespace riscv {

bool ConcatKernel::CanDoExecute(const KernelExecContext& ctx) const {
    bool all_empty = true;
    for (uint32_t i = 0; i < ctx.GetInputCount(); i++) {
        auto tensor = ctx.GetInput<TensorImpl>(i);
        if (!tensor) {
            return false;
        }
        if (tensor->GetShape()->CalcBytesIncludingPadding() != 0) {
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

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    for (uint32_t i = 0; i < ctx->GetInputCount(); ++i) {
        auto input = ctx->GetInput<TensorImpl>(i);
        PPLNN_RISCV_DEBUG_TRACE("Input [inputs[%u]]:\n", i);
        PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(input);
        src_shape_list_[i] = input->GetShape();
        src_list_[i] = input->GetBufferPtr();
    }
    PPLNN_RISCV_DEBUG_TRACE("Output [concat_result]:]n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(concat_result);
    PPLNN_RISCV_DEBUG_TRACE("axis: %d\n", param_->axis);

    auto data_type = concat_result->GetShape()->GetDataType();
    auto data_format = concat_result->GetShape()->GetDataFormat();
    const int32_t real_axis =
        param_->axis < 0 ? param_->axis + ctx->GetInput<TensorImpl>(0)->GetShape()->GetDimCount() : param_->axis;

    if ((ppl::common::GetSizeOfDataType(data_type) == 4 && real_axis == 1 &&
         data_format == ppl::common::DATAFORMAT_N4CX) ||
        (ppl::common::GetSizeOfDataType(data_type) == 2 && real_axis == 1 &&
         data_format == ppl::common::DATAFORMAT_N8CX)) {
        bool interleave_channels = false;
        int32_t c_blk = 0;
        if (data_format == ppl::common::DATAFORMAT_N8CX) {
            c_blk = 8;
        } else if (data_format == ppl::common::DATAFORMAT_N4CX) {
            c_blk = 4;
        }
        for (uint32_t i = 0; i < src_shape_list_.size() - 1; i++) {
            if (src_shape_list_[i]->GetDim(1) % c_blk != 0) {
                interleave_channels = true;
                break;
            }
        }
        if (interleave_channels) {
            if (data_format == ppl::common::DATAFORMAT_N8CX) {
                return kernel::riscv::concat_n8cx_interleave_channels_fp16(
                    (const __fp16**)src_list_.data(), concat_result->GetBufferPtr<__fp16>(),

                    src_shape_list_.data(), ctx->GetInputCount(), real_axis, 1);
            } else if (data_format == ppl::common::DATAFORMAT_N4CX) {
                return kernel::riscv::concat_n4cx_interleave_channels_fp32(
                    (const float**)src_list_.data(), concat_result->GetBufferPtr<float>(),

                    src_shape_list_.data(), ctx->GetInputCount(), real_axis, 1);
            }
        }
    }

    if (ppl::common::GetSizeOfDataType(data_type) == 2) {
        if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
            return kernel::riscv::concat_ndarray_fp16((const __fp16**)src_list_.data(),
                                                      concat_result->GetBufferPtr<__fp16>(), src_shape_list_.data(),
                                                      ctx->GetInputCount(), param_->axis);
        } else if (data_format == ppl::common::DATAFORMAT_N8CX) {
            return kernel::riscv::concat_n8cx_fp16((const __fp16**)src_list_.data(),
                                                   concat_result->GetBufferPtr<__fp16>(), src_shape_list_.data(),
                                                   ctx->GetInputCount(), param_->axis);
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format);
        }
    } else if (ppl::common::GetSizeOfDataType(data_type) == 4) {
        if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
            return kernel::riscv::concat_ndarray_fp32((const float**)src_list_.data(),
                                                      concat_result->GetBufferPtr<float>(), src_shape_list_.data(),
                                                      ctx->GetInputCount(), param_->axis);
        } else if (data_format == ppl::common::DATAFORMAT_N4CX) {
            return kernel::riscv::concat_n4cx_fp32((const float**)src_list_.data(),
                                                   concat_result->GetBufferPtr<float>(), src_shape_list_.data(),
                                                   ctx->GetInputCount(), param_->axis);
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format);
        }
    } else if (ppl::common::GetSizeOfDataType(data_type) == 8) {
        if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
            return kernel::riscv::concat_ndarray_int64((const int64_t**)src_list_.data(),
                                                       concat_result->GetBufferPtr<int64_t>(), src_shape_list_.data(),
                                                       ctx->GetInputCount(), param_->axis);
        } else if (data_format == ppl::common::DATAFORMAT_N2CX) {
            return kernel::riscv::concat_n2cx_int64((const int64_t**)src_list_.data(),
                                                    concat_result->GetBufferPtr<int64_t>(), src_shape_list_.data(),
                                                    ctx->GetInputCount(), param_->axis);
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format);
        }
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type);
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}}; // namespace ppl::nn::riscv
