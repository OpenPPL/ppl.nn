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

#include "ppl/nn/engines/x86/kernels/onnx/clip_kernel.h"
#include "ppl/kernel/x86/fp32/clip.h"
#include <algorithm>
#include <float.h>

namespace ppl { namespace nn { namespace x86 {

bool ClipKernel::CanDoExecute(const KernelExecContext& ctx) const {
    auto tensor = ctx.GetInput<TensorImpl>(0);
    if (!tensor || tensor->GetShape().GetBytesIncludingPadding() == 0) {
        return false;
    }
    return true;
}

ppl::common::RetCode ClipKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [input]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = input->GetShape().GetDataType();
    if (data_type != ppl::common::DATATYPE_FLOAT32) {
        LOG(ERROR) << "only support fp32 now.";
        return ppl::common::RC_UNSUPPORTED;
    }

    float min_val = -FLT_MAX;
    float max_val = FLT_MAX;
    if (ctx->GetInputCount() >= 2 && ctx->GetInput<TensorImpl>(1) != nullptr) {
        min_val = (ctx->GetInput<TensorImpl>(1)->GetBufferPtr<float>())[0];
    }
    if (ctx->GetInputCount() >= 3 && ctx->GetInput<TensorImpl>(2) != nullptr) {
        max_val = (ctx->GetInput<TensorImpl>(2)->GetBufferPtr<float>())[0];
    }

    PPLNN_X86_DEBUG_TRACE("min: %f\n", min_val);
    PPLNN_X86_DEBUG_TRACE("max: %f\n", max_val);

    if (MayUseISA(ppl::common::ISA_X86_AVX)) {
        return ppl::kernel::x86::clip_fp32_avx(&input->GetShape(), input->GetBufferPtr<float>(), min_val, max_val,
                                               output->GetBufferPtr<float>());
    } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
        return ppl::kernel::x86::clip_fp32_sse(&input->GetShape(), input->GetBufferPtr<float>(), min_val, max_val,
                                               output->GetBufferPtr<float>());
    } else {
        LOG(ERROR) << "get unsupported isa " << GetISA();
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
