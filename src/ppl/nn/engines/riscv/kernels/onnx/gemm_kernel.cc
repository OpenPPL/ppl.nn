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

#include "ppl/nn/engines/riscv/kernels/onnx/gemm_kernel.h"
#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/runtime/tensor_impl.h"

namespace ppl { namespace nn { namespace riscv {

ppl::common::RetCode GemmKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_RISCV_REQUIRED_INPUT(A, 0);
    PPLNN_RISCV_REQUIRED_INPUT(B, 1);
    PPLNN_RISCV_OPTIONAL_INPUT(C, 2);
    PPLNN_RISCV_REQUIRED_OUTPUT(Y, 0);

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [A]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(A);
    PPLNN_RISCV_DEBUG_TRACE("Input [B]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(B);
    if (C) {
        PPLNN_RISCV_DEBUG_TRACE("Input [C]:\n");
        PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(C);
    }
    PPLNN_RISCV_DEBUG_TRACE("Output [Y]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(Y);
    PPLNN_RISCV_DEBUG_TRACE("trans_A: %d\n", param_->transA);
    PPLNN_RISCV_DEBUG_TRACE("trans_B: %d\n", param_->transB);
    PPLNN_RISCV_DEBUG_TRACE("alpha: %f\n", param_->alpha);
    PPLNN_RISCV_DEBUG_TRACE("beta: %f\n", param_->beta);

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::riscv