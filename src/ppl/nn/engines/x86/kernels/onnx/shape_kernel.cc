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

#include "ppl/nn/engines/x86/kernels/onnx/shape_kernel.h"

namespace ppl { namespace nn { namespace x86 {

bool ShapeKernel::CanDoExecute(const KernelExecContext&) const {
    return true;
}

ppl::common::RetCode ShapeKernel::DoExecute(KernelExecContext* ctx) {
    auto data = ctx->GetInput<TensorImpl>(0);
    auto shape = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [data]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(data);
    PPLNN_X86_DEBUG_TRACE("Output [shape]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(shape);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    for (size_t i = 0; i < data->GetShape().GetRealDimCount(); i++) {
        shape->GetBufferPtr<int64_t>()[i] = data->GetShape().GetDim(i);
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::x86
