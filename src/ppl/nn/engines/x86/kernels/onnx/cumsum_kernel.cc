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

#include "ppl/nn/engines/x86/kernels/onnx/cumsum_kernel.h"
#include "ppl/kernel/x86/fp32/cumsum.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode CumSumKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(x, 0);
    PPLNN_X86_REQUIRED_INPUT(axis, 1);
    PPLNN_X86_REQUIRED_OUTPUT(y, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());

    PPLNN_X86_DEBUG_TRACE("Input [x]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_X86_DEBUG_TRACE("Input [axis]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(axis);

    PPLNN_X86_DEBUG_TRACE("exclusive: %d\n", param_->exclusive);
    PPLNN_X86_DEBUG_TRACE("reverse: %d\n", param_->reverse);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    PPLNN_X86_REALLOC_TENSOR_BUFFER(y);
    PPLNN_X86_DEBUG_TRACE("Output [y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(y);

    if (axis->GetShape()->GetDataType() != ppl::common::DATATYPE_INT64 &&
        axis->GetShape()->GetDataType() != ppl::common::DATATYPE_INT32) {
        LOG(ERROR) << "unsupported axis datatype: " << ppl::common::GetDataTypeStr(axis->GetShape()->GetDataType()) << ".";
    }

    const auto data_type = x->GetShape()->GetDataType();
    const int64_t axis_val =
        axis->GetShape()->GetDataType() == ppl::common::DATATYPE_INT64 ?
        axis->GetBufferPtr<const int64_t>()[0] :
        axis->GetBufferPtr<const int32_t>()[0];

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        return ppl::kernel::x86::cumsum_ndarray_fp32(
            x->GetShape(),
            x->GetBufferPtr<const float>(),
            axis_val,
            param_->exclusive,
            param_->reverse,
            y->GetBufferPtr<float>());
    } else {
        LOG(ERROR) << "unsupported x datatype: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
