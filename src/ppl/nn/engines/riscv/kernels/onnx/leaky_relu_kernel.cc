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

#include "ppl/nn/engines/riscv/kernels/onnx/leaky_relu_kernel.h"
#include "ppl/kernel/riscv/fp16/leaky_relu.h"
#include "ppl/kernel/riscv/fp32/leaky_relu.h"

namespace ppl { namespace nn { namespace riscv {

ppl::common::RetCode LeakyReLUKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_RISCV_REQUIRED_INPUT(x, 0);
    PPLNN_RISCV_REQUIRED_OUTPUT(y, 0);

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [x]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(x);

    PPLNN_RISCV_DEBUG_TRACE("alpha: %f\n", param_->alpha);

    PPLNN_RISCV_DEBUG_TRACE("Output [y]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(y);

    const auto data_type = x->GetShape()->GetDataType();

    if (ppl::common::DATATYPE_FLOAT16 == data_type) {
        return kernel::riscv::leaky_relu_n8cx_fp16(
            x->GetShape(),
            x->GetBufferPtr<__fp16>(),
            param_->alpha,
            y->GetBufferPtr<__fp16>()
        );
    } else if (ppl::common::DATATYPE_FLOAT32 == data_type) {
        return kernel::riscv::leaky_relu_n4cx_fp32(
            x->GetShape(),
            x->GetBufferPtr<float>(),
            param_->alpha,
            y->GetBufferPtr<float>()
        );
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type);
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::riscv
