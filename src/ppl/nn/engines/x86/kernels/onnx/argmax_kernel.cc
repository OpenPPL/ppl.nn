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

#include "ppl/nn/engines/x86/kernels/onnx/argmax_kernel.h"
#include "ppl/kernel/x86/fp32/argmax.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode ArgMaxKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(data, 0);
    PPLNN_X86_REQUIRED_OUTPUT(reduced, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());

    PPLNN_X86_DEBUG_TRACE("Input [data]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(data);

    PPLNN_X86_DEBUG_TRACE("axis: %d\n", param_->axis);
    PPLNN_X86_DEBUG_TRACE("keepdims: %d\n", param_->keepdims);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    PPLNN_X86_REALLOC_TENSOR_BUFFER(reduced);
    PPLNN_X86_DEBUG_TRACE("Output [reduced]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(reduced);

    const auto data_type = data->GetShape()->GetDataType();

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        return kernel::x86::argmax_ndarray_fp32(data->GetShape(), data->GetBufferPtr<float>(), param_->axis,
                                                reduced->GetBufferPtr<int64_t>());
    } else {
        LOG(ERROR) << "unsupported datatype: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
