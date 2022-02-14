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
// under the License

#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"

#include "ppl/nn/engines/riscv/kernels/onnx/not_kernel.h"
#include "ppl/kernel/riscv/bool/not.h"

namespace ppl { namespace nn { namespace riscv {

ppl::common::RetCode NotKernel::DoExecute(KernelExecContext* ctx) {
    auto src = ctx->GetInput<TensorImpl>(0);
    auto dst = ctx->GetOutput<TensorImpl>(0);

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [src]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(src);
    PPLNN_RISCV_DEBUG_TRACE("Output [dst]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(dst);

    const auto data_type = src->GetShape()->GetDataType();
    const auto data_format = src->GetShape()->GetDataFormat();
    return ppl::kernel::riscv::not_bool(src->GetShape(), src->GetBufferPtr<uint8_t>(), dst->GetBufferPtr<uint8_t>());

    return ppl::common::RC_UNSUPPORTED;
}

}}}; // namespace ppl::nn::riscv