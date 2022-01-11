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

#include "ppl/nn/engines/arm/kernels/onnx/reduce_prod_kernel.h"
#include "ppl/kernel/arm_server/reduce/neon/reduce.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode ReduceProdKernel::DoExecute(KernelExecContext* ctx) {
    auto data = ctx->GetInput<TensorImpl>(0);
    auto reduced = ctx->GetOutput<TensorImpl>(0);

    const uint32_t dim_count = data->GetShape()->GetDimCount();
    auto fixed_axes = param_->axes;
    if (param_->axes.empty()) { // empty axes means reduce all dims
        fixed_axes.resize(dim_count);
        for (size_t i = 0; i < dim_count; i++) {
            fixed_axes[i] = i;
        }
    }

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [data]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(data);
    PPLNN_ARM_DEBUG_TRACE("Input [reduced]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(reduced);
    for (uint32_t i = 0; i < fixed_axes.size(); ++i) {
        PPLNN_ARM_DEBUG_TRACE("axes[%d]: %d\n", i, fixed_axes[i]);
    }
    PPLNN_ARM_DEBUG_TRACE("keepdims: %d\n", param_->keep_dims);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    return ppl::kernel::arm_server::neon::reduce_prod(data->GetShape(), reduced->GetShape(),
                                                      data->GetBufferPtr<void>(), fixed_axes.data(), fixed_axes.size(),
                                                      reduced->GetBufferPtr<void>());
}

}}} // namespace ppl::nn::arm
