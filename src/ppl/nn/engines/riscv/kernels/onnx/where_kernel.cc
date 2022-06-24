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

#include "ppl/nn/engines/riscv/kernels/onnx/where_kernel.h"
#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/common/logger.h"
#include "ppl/kernel/riscv/fp32/where.h"
#include "ppl/kernel/riscv/fp16/where.h"
#include "ppl/kernel/riscv/int64/where.h"

namespace ppl { namespace nn { namespace riscv {

ppl::common::RetCode WhereKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_RISCV_REQUIRED_INPUT(cond, 0);
    PPLNN_RISCV_REQUIRED_INPUT(x, 1);
    PPLNN_RISCV_REQUIRED_INPUT(y, 2);
    PPLNN_RISCV_REQUIRED_OUTPUT(output, 0);

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [cond]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_RISCV_DEBUG_TRACE("Input [x]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_RISCV_DEBUG_TRACE("Input [y]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(y);

    // PPLNN_RISCV_REALLOC_TENSOR_BUFFER(output);
    PPLNN_RISCV_DEBUG_TRACE("Output [output]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(output);

    bool is_eltwise =
        cond->GetShape()->CalcElementsIncludingPadding() == output->GetShape()->CalcElementsIncludingPadding() &&
        x->GetShape()->CalcElementsIncludingPadding() == output->GetShape()->CalcElementsIncludingPadding() &&
        y->GetShape()->CalcElementsIncludingPadding() == output->GetShape()->CalcElementsIncludingPadding();

    if (output->GetShape()->GetDataType() == common::DATATYPE_FLOAT32) {
        if (is_eltwise) {
            return kernel::riscv::where_eltwise_fp32(output->GetShape(), cond->GetBufferPtr<const uint8_t>(),
                                                     x->GetBufferPtr<const float>(), y->GetBufferPtr<const float>(),
                                                     output->GetBufferPtr<float>());
        } else {
            return kernel::riscv::where_ndarray_fp32(
                cond->GetShape(), x->GetShape(), y->GetShape(), output->GetShape(), cond->GetBufferPtr<const uint8_t>(),
                x->GetBufferPtr<const float>(), y->GetBufferPtr<const float>(), output->GetBufferPtr<float>());
        }
    } else if (output->GetShape()->GetDataType() == common::DATATYPE_FLOAT16) {
        if (is_eltwise) {
            return kernel::riscv::where_eltwise_fp16(output->GetShape(), cond->GetBufferPtr<const uint8_t>(),
                                                     x->GetBufferPtr<const __fp16>(), y->GetBufferPtr<const __fp16>(),
                                                     output->GetBufferPtr<__fp16>());
        } else {
            return kernel::riscv::where_ndarray_fp16(
                cond->GetShape(), x->GetShape(), y->GetShape(), output->GetShape(), cond->GetBufferPtr<const uint8_t>(),
                x->GetBufferPtr<const __fp16>(), y->GetBufferPtr<const __fp16>(), output->GetBufferPtr<__fp16>());
        }
    } else if (output->GetShape()->GetDataType() == common::DATATYPE_INT64) {
        if (is_eltwise) {
            return kernel::riscv::where_eltwise_int64(
                output->GetShape(), cond->GetBufferPtr<const uint8_t>(), x->GetBufferPtr<const int64_t>(),
                y->GetBufferPtr<const int64_t>(), output->GetBufferPtr<int64_t>());
        } else {
            return kernel::riscv::where_ndarray_int64(
                cond->GetShape(), x->GetShape(), y->GetShape(), output->GetShape(), cond->GetBufferPtr<const uint8_t>(),
                x->GetBufferPtr<const int64_t>(), y->GetBufferPtr<const int64_t>(), output->GetBufferPtr<int64_t>());
        }
    } else {
        LOG(ERROR) << "unsupported data type " << common::GetDataTypeStr(output->GetShape()->GetDataType());
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::riscv