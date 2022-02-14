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

#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"

#include "ppl/nn/engines/riscv/kernels/mmcv/mmcv_gridsample_kernel.h"
#include "ppl/kernel/riscv/fp16/mmcv_gridsample.h"
#include "ppl/kernel/riscv/fp32/mmcv_gridsample.h"

namespace ppl { namespace nn { namespace riscv {

ppl::common::RetCode MMCVGridSampleKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto grid = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_RISCV_DEBUG_TRACE("Input [input]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_RISCV_DEBUG_TRACE("Input [grid]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(grid);

    PPLNN_RISCV_DEBUG_TRACE("align_corners: %ld\n", param_->align_corners);
    PPLNN_RISCV_DEBUG_TRACE("interpolation_mode: %ld\n", param_->interpolation_mode);
    PPLNN_RISCV_DEBUG_TRACE("padding_mode: %ld\n", param_->padding_mode);

    // PPLNN_RISCV_REALLOC_TENSOR_BUFFER(output);
    PPLNN_RISCV_DEBUG_TRACE("Output [output]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(output);

    const auto data_type = input->GetShape()->GetDataType();
    const auto data_format = input->GetShape()->GetDataFormat();
    if (0 == param_->interpolation_mode) { // bilinear
        if (ppl::common::DATATYPE_FLOAT16 == data_type && ppl::common::DATAFORMAT_N8CX == data_format) {
            return kernel::riscv::mmcv_gridsample_bilinear_n8cx_fp16(
                input->GetShape(), grid->GetShape(), input->GetBufferPtr<__fp16>(), grid->GetBufferPtr<float>(),
                param_->align_corners, param_->padding_mode, output->GetBufferPtr<__fp16>());
        } else if (ppl::common::DATATYPE_FLOAT32 == data_type && ppl::common::DATAFORMAT_N4CX == data_format) {
            return kernel::riscv::mmcv_gridsample_bilinear_n4cx_fp32(
                input->GetShape(), grid->GetShape(), input->GetBufferPtr<float>(), grid->GetBufferPtr<float>(),
                param_->align_corners, param_->padding_mode, output->GetBufferPtr<float>());
        }
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}}; // namespace ppl::nn::riscv