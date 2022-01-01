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

    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    PPLNN_X86_REALLOC_TENSOR_BUFFER(Y);
    PPLNN_X86_DEBUG_TRACE("Output [Y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);

    const auto data_type = X->GetShape()->GetDataType();
    const auto data_format = X->GetShape()->GetDataFormat();

    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        if (slope->GetShape()->GetDimCount() > X->GetShape()->GetDimCount()) {
            LOG(ERROR) << "prelu slope dimcount is bigger than input dimcount.";
            return ppl::common::RC_UNSUPPORTED;
        }
        auto channels = X->GetShape()->GetDimCount() > 1 ? X->GetShape()->GetDim(1) : 1;
        if (slope->GetShape()->GetElementsExcludingPadding() != (uint64_t)channels) {
            LOG(ERROR) << "prelu only support channel broadcasting.";
            return ppl::common::RC_UNSUPPORTED;
        }
        if (slope->GetShape()->GetDimCount() > 1 && channels > 1) {
            int32_t broadcast_dim = -1;
            for (uint32_t i = 0; i < slope->GetShape()->GetDimCount(); ++i) {
                if (slope->GetShape()->GetDim(i) == channels) {
                    broadcast_dim = i;
                    break;
                }
            }
            if (broadcast_dim == -1 ||
                slope->GetShape()->GetDimCount() - broadcast_dim != X->GetShape()->GetDimCount() - 1) {
                LOG(ERROR) << "prelu only support channel broadcasting.";
                return ppl::common::RC_UNSUPPORTED;
            }
        }
    } else {
        LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format) << ".";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        if (MayUseISA(ppl::common::ISA_X86_AVX)) {
            return kernel::x86::prelu_channel_shared_fp32_avx(X->GetShape(), X->GetBufferPtr<float>(),
                                                              slope->GetBufferPtr<float>(), Y->GetBufferPtr<float>());
        } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
            return kernel::x86::prelu_channel_shared_fp32_sse(X->GetShape(), X->GetBufferPtr<float>(),
                                                              slope->GetBufferPtr<float>(), Y->GetBufferPtr<float>());
        } else {
            LOG(ERROR) << "get unsupported isa " << GetISA();
        }
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
