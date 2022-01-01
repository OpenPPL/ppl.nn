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

#include "ppl/nn/engines/riscv/kernels/onnx/add_kernel.h"
#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"

#include "ppl/kernel/riscv/fp16/arithmetic.h"
#include "ppl/kernel/riscv/fp32/arithmetic.h"
#include "ppl/kernel/riscv/int64/arithmetic.h"

namespace ppl { namespace nn { namespace riscv {

ppl::common::RetCode AddKernel::DoExecute(KernelExecContext* ctx) {
    auto A = ctx->GetInput<TensorImpl>(0);
    auto B = ctx->GetInput<TensorImpl>(1);
    auto C = ctx->GetOutput<TensorImpl>(0);

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [A]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(A);
    PPLNN_RISCV_DEBUG_TRACE("Input [B]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(B);
    PPLNN_RISCV_DEBUG_TRACE("Output [C]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(C);

    const common::datatype_t input_data_type_0 = ctx->GetInput<TensorImpl>(0)->GetShape()->GetDataType();
    const common::datatype_t input_data_type_1 = ctx->GetInput<TensorImpl>(1)->GetShape()->GetDataType();
    const common::datatype_t output_data_type = ctx->GetOutput<TensorImpl>(0)->GetShape()->GetDataType();

    if (input_data_type_0 != input_data_type_1 || input_data_type_0 != output_data_type) {
        return ppl::common::RC_UNSUPPORTED;
    }

    if (output_data_type == common::DATATYPE_FLOAT16) {
        return kernel::riscv::add_fp16(A->GetShape(), B->GetShape(), C->GetShape(), false,
                                       A->GetBufferPtr<const __fp16>(), B->GetBufferPtr<const __fp16>(),
                                       C->GetBufferPtr<__fp16>());
    } else if (output_data_type == common::DATATYPE_FLOAT32) {
        return kernel::riscv::add_fp32(A->GetShape(), B->GetShape(), C->GetShape(), false,
                                       A->GetBufferPtr<const float>(), B->GetBufferPtr<const float>(),
                                       C->GetBufferPtr<float>());
    } else if (output_data_type == common::DATATYPE_INT64) {
        return kernel::riscv::add_int64(A->GetShape(), B->GetShape(), C->GetShape(), false,
                                        A->GetBufferPtr<const int64_t>(), B->GetBufferPtr<const int64_t>(),
                                        C->GetBufferPtr<int64_t>());
    } else {
        LOG(ERROR) << "unsupported datatype " << common::GetDataTypeStr(output_data_type);
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::riscv
