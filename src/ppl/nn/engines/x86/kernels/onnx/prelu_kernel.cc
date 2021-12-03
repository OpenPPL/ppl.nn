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

#include "ppl/nn/engines/x86/kernels/onnx/prelu_kernel.h"
#include "ppl/kernel/x86/fp32/prelu.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode PReluKernel::DoExecute(KernelExecContext* ctx) {

    PPLNN_X86_REQUIRED_INPUT(X, 0);
    PPLNN_X86_REQUIRED_INPUT(slope, 1);
    PPLNN_X86_REQUIRED_OUTPUT(Y, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [X]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(X);
    PPLNN_X86_DEBUG_TRACE("Input [slope]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(slope);
    PPLNN_X86_DEBUG_TRACE("Input [Y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    auto x = ctx->GetInput<TensorImpl>(0);
    auto y = ctx->GetOutput<TensorImpl>(0);

    const auto data_type = x->GetShape().GetDataType();
    const auto data_format = X->GetShape().GetDataFormat();

    if(data_format != ppl::common::DATAFORMAT_NDARRAY){
        LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format) << "."; 
        return ppl::common::RC_UNSUPPORTED;
    }

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        if (MayUseISA(ppl::common::ISA_X86_AVX)) {
            return kernel::x86::prelu_fp32_avx(&x->GetShape(), x->GetBufferPtr<float>(), 
                                                    slope->GetBufferPtr<float>(),
                                                    y->GetBufferPtr<float>());
        } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
            return kernel::x86::prelu_fp32_sse(&x->GetShape(), x->GetBufferPtr<float>(), 
                                                    slope->GetBufferPtr<float>(),
                                                    y->GetBufferPtr<float>());
        } else {
            LOG(ERROR) << "get unsupported isa " << GetISA();
        }
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
