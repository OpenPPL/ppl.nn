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

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/runtime/tensor_impl.h"

#include "ppl/kernel/riscv/fp16/reduce.h"
#include "ppl/kernel/riscv/fp32/reduce.h"
#include "ppl/nn/engines/riscv/kernels/onnx/reduce_mean_kernel.h"

namespace ppl { namespace nn { namespace riscv {

ppl::common::RetCode ReduceMeanKernel::DoExecute(KernelExecContext* ctx) {
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

    for (uint32_t i = 0; i < fixed_axes.size(); i++) {
        if (fixed_axes[i] < 0) { // turn negative axes to positive axes
            fixed_axes[i] = fixed_axes[i] + dim_count;
        }
    }

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [data]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(data);
    PPLNN_RISCV_DEBUG_TRACE("Input [reduced]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(reduced);
    for (uint32_t i = 0; i < fixed_axes.size(); ++i) {
        PPLNN_RISCV_DEBUG_TRACE("axes[%d]: %d\n", i, fixed_axes[i]);
    }
    PPLNN_RISCV_DEBUG_TRACE("keepdims: %d\n", param_->keep_dims);

    auto data_type = data->GetShape()->GetDataType();
    if (data_type == ppl::common::DATATYPE_FLOAT16) {
        return kernel::riscv::reduce_mean_fp16(data->GetBufferPtr<__fp16>(), reduced->GetBufferPtr<__fp16>(),
                                               data->GetShape(), reduced->GetShape(), fixed_axes.data(),
                                               fixed_axes.size());
    } else if (data_type == ppl::common::DATATYPE_FLOAT32) {
        return kernel::riscv::reduce_mean_fp32(data->GetBufferPtr<float>(), reduced->GetBufferPtr<float>(),
                                               data->GetShape(), reduced->GetShape(), fixed_axes.data(),
                                               fixed_axes.size());
    } else if (data_type == ppl::common::DATATYPE_INT64) {
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::riscv
