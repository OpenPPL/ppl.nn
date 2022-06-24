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

#include "ppl/nn/engines/riscv/kernels/onnx/flatten_kernel.h"
#include "ppl/kernel/riscv/common/memory.h"
#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/common/log.h"
#include "ppl/kernel/riscv/fp16/flatten.h"
#include "ppl/kernel/riscv/fp32/flatten.h"

namespace ppl { namespace nn { namespace riscv {

ppl::common::RetCode FlattenKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [input]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_RISCV_DEBUG_TRACE("Output [output]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_RISCV_DEBUG_TRACE("axis: %d\n", param_->axis);

    auto input_data_format = input->GetShape()->GetDataFormat();
    const int64_t size_2D = input->GetShape()->GetDim(2) * input->GetShape()->GetDim(3);

    if (input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL &&
        ppl::common::DATAFORMAT_NDARRAY == input_data_format) {
        output->TransferBufferFrom(input);
    } else if (size_2D != 1 && ppl::common::DATAFORMAT_N8CX == input_data_format) {
        return ppl::kernel::riscv::flatten_n8cx_fp16(input->GetBufferPtr<__fp16>(), output->GetBufferPtr<__fp16>(),
                                                     input->GetShape(), output->GetShape());
    } else if (size_2D != 1 && ppl::common::DATAFORMAT_N4CX == input_data_format) {
        return ppl::kernel::riscv::flatten_n4cx_fp32(input->GetBufferPtr<float>(), output->GetBufferPtr<float>(),
                                                     input->GetShape(), output->GetShape());
    } else {
        return ppl::kernel::riscv::memory_copy(input->GetBufferPtr(), input->GetShape()->CalcBytesIncludingPadding(),
                                               output->GetBufferPtr());
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::riscv
