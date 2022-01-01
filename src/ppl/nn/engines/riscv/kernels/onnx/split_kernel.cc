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

#include "ppl/nn/engines/riscv/kernels/onnx/split_kernel.h"
#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/runtime/tensor_impl.h"

#include "ppl/kernel/riscv/fp32/split.h"
#include "ppl/kernel/riscv/fp16/split.h"

namespace ppl { namespace nn { namespace riscv {

ppl::common::RetCode SplitKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);

    std::vector<void*> dst_list(ctx->GetOutputCount());
    std::vector<const TensorShape*> dst_shape_list(ctx->GetOutputCount());

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [input]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(input);
    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto output = ctx->GetOutput<TensorImpl>(i);
        PPLNN_RISCV_DEBUG_TRACE("Output [outputs[%u]]:\n", i);
        PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(output);
        dst_list[i] = output->GetBufferPtr<void>();
        dst_shape_list[i] = output->GetShape();
    }
    PPLNN_RISCV_DEBUG_TRACE("axis: %d\n", param_->axis);

    const int32_t real_axis =
        param_->axis < 0 ? param_->axis + ctx->GetInput<TensorImpl>(0)->GetShape()->GetDimCount() : param_->axis;

    auto data_type = input->GetShape()->GetDataType();
    auto data_format = input->GetShape()->GetDataFormat();
    if (ppl::common::GetSizeOfDataType(data_type) == 4) {
        if (data_format == ppl::common::DATAFORMAT_N4CX) {
            return kernel::riscv::split_n4cx_fp32(input->GetShape(), dst_shape_list.data(),
                                                  input->GetBufferPtr<float>(), param_->axis, ctx->GetOutputCount(),
                                                  (float**)dst_list.data());
        } else if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
            return kernel::riscv::split_ndarray_fp32(input->GetShape(), dst_shape_list.data(),
                                                     input->GetBufferPtr<float>(), param_->axis, ctx->GetOutputCount(),
                                                     (float**)dst_list.data());
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format);
        }
    } else if (ppl::common::GetSizeOfDataType(data_type) == 2) {
        if (data_format == ppl::common::DATAFORMAT_N8CX) {
            return kernel::riscv::split_n8cx_fp16(input->GetShape(), dst_shape_list.data(),
                                                  input->GetBufferPtr<__fp16>(), param_->axis, ctx->GetOutputCount(),
                                                  (__fp16**)dst_list.data());
        } else if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
            return kernel::riscv::split_ndarray_fp16(input->GetShape(), dst_shape_list.data(),
                                                     input->GetBufferPtr<__fp16>(), param_->axis, ctx->GetOutputCount(),
                                                     (__fp16**)dst_list.data());
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format);
        }
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::riscv
