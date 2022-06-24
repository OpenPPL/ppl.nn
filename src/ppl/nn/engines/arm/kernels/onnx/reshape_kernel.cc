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

#include <string.h>

#include "ppl/nn/engines/arm/kernels/onnx/reshape_kernel.h"
#include "ppl/kernel/arm_server/common/memory.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode ReshapeKernel::DoExecute(KernelExecContext* ctx) {
    auto data = ctx->GetInput<TensorImpl>(0);
    auto shape = ctx->GetInput<TensorImpl>(1);
    auto reshaped = ctx->GetOutput<TensorImpl>(0);

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [data]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(data);
    PPLNN_ARM_DEBUG_TRACE("Input [shape]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(shape);
    PPLNN_ARM_DEBUG_TRACE("Output [reshaped]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(reshaped);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    if (data->GetEdge()->CalcConsumerCount() == 1 && data->GetType() == TENSORTYPE_NORMAL) {
        reshaped->TransferBufferFrom(data);
    } else {
        ppl::kernel::arm_server::memory_copy(data->GetBufferPtr(), data->GetShape()->CalcBytesIncludingPadding(),
                                             reshaped->GetBufferPtr());
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::arm
