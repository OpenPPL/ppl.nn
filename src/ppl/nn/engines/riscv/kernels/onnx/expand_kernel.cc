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

#include "ppl/nn/engines/riscv/kernels/onnx/expand_kernel.h"
#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/kernel/riscv/common/expand.h"

namespace ppl { namespace nn { namespace riscv {

bool ExpandKernel::CanDoExecute(const KernelExecContext& ctx) const {
    auto input = ctx.GetInput<TensorImpl>(0);
    auto shape = ctx.GetInput<TensorImpl>(1);
    auto output = ctx.GetOutput<TensorImpl>(0);
    if (input->GetShape()->CalcBytesIncludingPadding() == 0 || shape->GetShape()->CalcBytesIncludingPadding() == 0 ||
        output->GetShape()->CalcBytesIncludingPadding() == 0) {
        return false;
    }
    return true;
}

ppl::common::RetCode ExpandKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_RISCV_REQUIRED_INPUT(input, 0);
    PPLNN_RISCV_REQUIRED_INPUT(shape, 1);
    PPLNN_RISCV_REQUIRED_OUTPUT(output, 0);

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());

    PPLNN_RISCV_DEBUG_TRACE("Input [input]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_RISCV_DEBUG_TRACE("Input [shape]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(shape);

    // PPLNN_RISCV_REALLOC_TENSOR_BUFFER(output);
    PPLNN_RISCV_DEBUG_TRACE("Output [output]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(output);

    return ppl::kernel::riscv::expand(input->GetShape(), output->GetShape(), input->GetBufferPtr<void>(),
                                      output->GetBufferPtr<void>());
}

}}} // namespace ppl::nn::riscv