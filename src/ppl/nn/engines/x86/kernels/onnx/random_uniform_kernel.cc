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

#include <math.h>

#include "ppl/nn/engines/x86/kernels/onnx/random_uniform_kernel.h"
#include "ppl/kernel/x86/fp32/random.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode RandomUniformKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_OUTPUT(output, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());

    PPLNN_X86_DEBUG_TRACE("dtype: %d\n", param_->dtype);
    PPLNN_X86_DEBUG_TRACE("high: %f\n", param_->high);
    PPLNN_X86_DEBUG_TRACE("low: %f\n", param_->low);
    PPLNN_X86_DEBUG_TRACE("seed: %f\n", param_->seed.size() ? param_->seed[0] : NAN);
    PPLNN_X86_DEBUG_TRACE("shape: (");
    for (uint64_t i = 0; i < param_->shape.size(); ++i) {
        PPLNN_X86_DEBUG_TRACE("%d, ", param_->shape[i]);
    }
    PPLNN_X86_DEBUG_TRACE(")\n");
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    PPLNN_X86_REALLOC_TENSOR_BUFFER(output);

    PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);

    const auto data_type = output->GetShape()->GetDataType();

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        return kernel::x86::random_uniform_fp32(
            output->GetShape(),
            param_->seed.size() ? param_->seed.data() : nullptr,
            param_->high, param_->low,
            output->GetBufferPtr<float>());
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
