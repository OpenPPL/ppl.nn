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

#include "ppl/nn/engines/riscv/kernels/onnx/relu_kernel.h"
#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/runtime/tensor_impl.h"

#include "ppl/kernel/riscv/fp16/relu.h"
#include "ppl/kernel/riscv/fp32/relu.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

RetCode ReluKernel::DoExecute(KernelExecContext* ctx) {
    auto X = ctx->GetInput<TensorImpl>(0);
    auto Y = ctx->GetOutput<TensorImpl>(0);

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [X]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(X);
    PPLNN_RISCV_DEBUG_TRACE("Output [Y]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(Y);

    const auto data_type = X->GetShape()->GetDataType();

    if (data_type == ppl::common::DATATYPE_FLOAT16) {
        return kernel::riscv::relu_fp16(X->GetShape(), X->GetBufferPtr<const __fp16>(), Y->GetBufferPtr<__fp16>());
    } else if (data_type == ppl::common::DATATYPE_FLOAT32) {
        return kernel::riscv::relu_fp32(X->GetShape(), X->GetBufferPtr<const float>(), Y->GetBufferPtr<float>());
    } else {
        LOG(ERROR) << "unsupported datatype: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::riscv
