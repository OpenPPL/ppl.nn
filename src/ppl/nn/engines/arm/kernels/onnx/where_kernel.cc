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

#include "ppl/nn/engines/arm/kernels/onnx/where_kernel.h"
#include "ppl/kernel/arm_server/where/neon/where.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode WhereKernel::DoExecute(KernelExecContext* ctx) {
    auto cond = ctx->GetInput<TensorImpl>(0);
    auto x = ctx->GetInput<TensorImpl>(1);
    auto y = ctx->GetInput<TensorImpl>(2);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [cond]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_ARM_DEBUG_TRACE("Input [x]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_ARM_DEBUG_TRACE("Input [y]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(y);
    PPLNN_ARM_DEBUG_TRACE("Output [output]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    return ppl::kernel::arm_server::neon::where(cond->GetShape(), x->GetShape(), y->GetShape(), output->GetShape(),
                                                cond->GetBufferPtr<void>(), x->GetBufferPtr<void>(),
                                                y->GetBufferPtr<void>(), output->GetBufferPtr<void>());
}

}}} // namespace ppl::nn::arm
