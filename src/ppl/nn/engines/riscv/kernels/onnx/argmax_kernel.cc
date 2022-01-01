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

#include "ppl/nn/engines/riscv/kernels/onnx/argmax_kernel.h"
#include "ppl/kernel/riscv/fp32/argmax.h"
#include "ppl/kernel/riscv/fp16/argmax.h"

namespace ppl { namespace nn { namespace riscv {

ppl::common::RetCode ArgMaxKernel::DoExecute(KernelExecContext* ctx) {
    auto data = ctx->GetInput<TensorImpl>(0);
    auto reduced = ctx->GetOutput<TensorImpl>(0);

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [data]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(data);
    PPLNN_RISCV_DEBUG_TRACE("Input [reduced]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(reduced);
    PPLNN_RISCV_DEBUG_TRACE("axis: %d\n", param_->axis);
    PPLNN_RISCV_DEBUG_TRACE("keepdims: %d\n", param_->keepdims);

    const auto data_type = data->GetShape()->GetDataType();

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        return kernel::riscv::argmax_ndarray_fp32(data->GetShape(), data->GetBufferPtr<float>(), param_->axis,
                                                  reduced->GetBufferPtr<int64_t>());
    } else if (data_type == ppl::common::DATATYPE_FLOAT16) {
        return kernel::riscv::argmax_ndarray_fp16(data->GetShape(), data->GetBufferPtr<__fp16>(), param_->axis,
                                                  reduced->GetBufferPtr<int64_t>());
    } else {
        LOG(ERROR) << "unsupported datatype: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::riscv
