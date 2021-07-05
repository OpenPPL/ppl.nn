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

#include "ppl/nn/engines/x86/kernels/onnx/identity_kernel.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode IdentityKernel::DoExecute(KernelExecContext* ctx) {
    auto input_tensor = ctx->GetInput<TensorImpl>(0);
    auto output_tensor = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("identity_kernel bottom: %p top: %p\n", input_tensor, output_tensor);
    PPLNN_X86_DEBUG_TRACE("name: %s\n", input_tensor->GetName());
    PPLNN_X86_DEBUG_TRACE("ptr: %p\n", input_tensor->GetBufferPtr());
    PPLNN_X86_DEBUG_TRACE("dims: %d %d %d %d\n", (int)input_tensor->GetShape().GetDim(0),
                          (int)input_tensor->GetShape().GetDim(1), (int)input_tensor->GetShape().GetDim(2),
                          (int)input_tensor->GetShape().GetDim(3));
    PPLNN_X86_DEBUG_TRACE("dataType: %s\n", ppl::common::GetDataTypeStr(input_tensor->GetShape().GetDataType()));
    PPLNN_X86_DEBUG_TRACE("dataFormat: %s\n", ppl::common::GetDataFormatStr(input_tensor->GetShape().GetDataFormat()));
    PPLNN_X86_DEBUG_TRACE("name: %s\n", output_tensor->GetName());
    PPLNN_X86_DEBUG_TRACE("ptr: %p\n", output_tensor->GetBufferPtr());
    PPLNN_X86_DEBUG_TRACE("dims: %d %d %d %d\n", (int)output_tensor->GetShape().GetDim(0),
                          (int)output_tensor->GetShape().GetDim(1), (int)output_tensor->GetShape().GetDim(2),
                          (int)output_tensor->GetShape().GetDim(3));
    PPLNN_X86_DEBUG_TRACE("dataType: %s\n", ppl::common::GetDataTypeStr(output_tensor->GetShape().GetDataType()));
    PPLNN_X86_DEBUG_TRACE("dataFormat: %s\n", ppl::common::GetDataFormatStr(output_tensor->GetShape().GetDataFormat()));

    memcpy(output_tensor->GetBufferPtr(), input_tensor->GetBufferPtr(),
           input_tensor->GetShape().GetBytesIncludingPadding());

    return 0;
}

}}} // namespace ppl::nn::x86
