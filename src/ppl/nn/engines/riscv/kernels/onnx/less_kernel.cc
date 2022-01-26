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

#include "ppl/nn/engines/riscv/kernels/onnx/less_kernel.h"
#include "ppl/kernel/riscv/fp16/relation.h"
#include "ppl/kernel/riscv/fp32/relation.h"
#include "ppl/kernel/riscv/int64/relation.h"

namespace ppl { namespace nn { namespace riscv {

ppl::common::RetCode LessKernel::DoExecute(KernelExecContext* ctx) {
    auto X = ctx->GetInput<TensorImpl>(0);
    auto Y = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [X]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(X);
    PPLNN_RISCV_DEBUG_TRACE("Input [Y]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(Y);
    PPLNN_RISCV_DEBUG_TRACE("Output [output]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(output);

    const auto data_type = X->GetShape()->GetDataType();
    if (data_type == ppl::common::DATATYPE_FLOAT16) {
        return kernel::riscv::less_fp16(
            X->GetShape(),
            Y->GetShape(),
            output->GetShape(),
            X->GetBufferPtr<__fp16>(),
            Y->GetBufferPtr<__fp16>(),
            output->GetBufferPtr<uint8_t>()
        );
    } else if (data_type == ppl::common::DATATYPE_FLOAT32) {
        return kernel::riscv::less_fp32(
            X->GetShape(),
            Y->GetShape(),
            output->GetShape(),
            X->GetBufferPtr<float>(),
            Y->GetBufferPtr<float>(),
            output->GetBufferPtr<uint8_t>()
        );
    } else if (data_type == ppl::common::DATATYPE_INT64) {
        return kernel::riscv::less_int64(
            X->GetShape(),
            Y->GetShape(),
            output->GetShape(),
            X->GetBufferPtr<int64_t>(),
            Y->GetBufferPtr<int64_t>(),
            output->GetBufferPtr<uint8_t>()
        );
    } else {
        LOG(ERROR) << "unsupported datatype " << common::GetDataTypeStr(data_type);
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}}