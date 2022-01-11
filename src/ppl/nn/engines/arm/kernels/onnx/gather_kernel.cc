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

#include <vector>

#include "ppl/nn/engines/arm/kernels/onnx/gather_kernel.h"
#include "ppl/kernel/arm_server/gather/neon/gather.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode GatherKernel::DoExecute(KernelExecContext* ctx) {
    auto x = ctx->GetInput<TensorImpl>(0);
    auto indices = ctx->GetInput<TensorImpl>(1);
    auto y = ctx->GetOutput<TensorImpl>(0);

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [x]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_ARM_DEBUG_TRACE("Input [indices]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(indices);
    PPLNN_ARM_DEBUG_TRACE("Output [y]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(y);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    const int64_t real_axis = param_->axis < 0 ? param_->axis + x->GetShape()->GetDimCount() : param_->axis;

    return ppl::kernel::arm_server::neon::gather(x->GetShape(), indices->GetShape(), x->GetBufferPtr<void>(),
                                                 indices->GetBufferPtr<int64_t>(), real_axis, y->GetBufferPtr<void>());
}

}}} // namespace ppl::nn::arm
