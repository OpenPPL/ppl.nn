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

#include "ppl/nn/engines/arm/kernels/onnx/transpose_kernel.h"
#include "ppl/kernel/arm_server/transpose/neon/transpose.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode TransposeKernel::DoExecute(KernelExecContext* ctx) {
    auto data = ctx->GetInput<TensorImpl>(0);
    auto transposed = ctx->GetOutput<TensorImpl>(0);

    const uint32_t dim_count = data->GetShape()->GetDimCount();
    std::vector<int64_t> modified_perm(dim_count);
    if (param_->perm.empty()) { // perm is empty, default is reverse dimension.
        for (uint32_t i = 0; i < dim_count; i++) {
            modified_perm[i] = dim_count - i - 1;
        }
    } else {
        for (uint32_t i = 0; i < dim_count; i++) {
            modified_perm[i] = param_->perm[i] < 0 ? param_->perm[i] + dim_count : param_->perm[i];
        }
    }

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [data]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(data);
    PPLNN_ARM_DEBUG_TRACE("Output [transposed]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(transposed);
    for (uint32_t i = 0; i < data->GetShape()->GetDimCount(); ++i) {
        PPLNN_ARM_DEBUG_TRACE("perm[%u]: %ld\n", i, modified_perm[i]);
    }
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    return ppl::kernel::arm_server::neon::transpose(data->GetShape(), transposed->GetShape(), modified_perm.data(),
                                                    data->GetBufferPtr<void>(), transposed->GetBufferPtr<void>());
}

}}} // namespace ppl::nn::arm
