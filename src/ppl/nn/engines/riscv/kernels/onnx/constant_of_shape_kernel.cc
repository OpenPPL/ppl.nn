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

#include "ppl/nn/engines/riscv/kernels/onnx/constant_of_shape_kernel.h"
#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/kernel/riscv/common/memory.h"

#include <chrono>

namespace ppl { namespace nn { namespace riscv {

ppl::common::RetCode ConstantOfShapeKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_RISCV_REQUIRED_INPUT(input, 0);
    PPLNN_RISCV_REQUIRED_OUTPUT(output, 0);

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());

    PPLNN_RISCV_DEBUG_TRACE("Input [input]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(input);

    // PPLNN_RISCV_REALLOC_TENSOR_BUFFER(output);
    PPLNN_RISCV_DEBUG_TRACE("Output [output]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(output);

    uint64_t output_datatype_size = ppl::common::GetSizeOfDataType(output->GetShape()->GetDataType());
    return kernel::riscv::memory_init(param_->data.GetData(), output_datatype_size,
                                      output->GetShape()->CalcElementsIncludingPadding(), output->GetBufferPtr());
}

}}} // namespace ppl::nn::riscv
