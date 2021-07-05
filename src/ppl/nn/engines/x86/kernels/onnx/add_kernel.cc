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

#include "ppl/nn/engines/x86/kernels/onnx/add_kernel.h"
#include "ppl/nn/common/logger.h"

#include "ppl/kernel/x86/fp32/arithmetic.h"
#include "ppl/kernel/x86/int64/arithmetic.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode AddKernel::DoExecute(KernelExecContext* ctx) {
    auto A = ctx->GetInput<TensorImpl>(0);
    auto B = ctx->GetInput<TensorImpl>(1);
    auto C = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [A]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(A);
    PPLNN_X86_DEBUG_TRACE("Input [B]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(B);
    PPLNN_X86_DEBUG_TRACE("Output [C]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(C);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const common::datatype_t data_type = ctx->GetOutput<TensorImpl>(0)->GetShape().GetDataType();

    if (fuse_relu_ && data_type != common::DATATYPE_FLOAT32) {
        LOG(ERROR) << "fuse relu support DATATYPE_FLOAT32 only.";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (data_type == common::DATATYPE_FLOAT32) {
        if (MayUseISA(ppl::common::ISA_X86_AVX)) {
            return kernel::x86::add_fp32_avx(&A->GetShape(), &B->GetShape(), &C->GetShape(),
                                             A->GetBufferPtr<const float>(), B->GetBufferPtr<const float>(), fuse_relu_,
                                             C->GetBufferPtr<float>());
        } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
            return kernel::x86::add_fp32_sse(&A->GetShape(), &B->GetShape(), &C->GetShape(),
                                             A->GetBufferPtr<const float>(), B->GetBufferPtr<const float>(), fuse_relu_,
                                             C->GetBufferPtr<float>());
        } else {
            LOG(ERROR) << "get unsupported isa " << GetISA();
            return ppl::common::RC_UNSUPPORTED;
        }
    } else if (data_type == common::DATATYPE_INT64) {
        return kernel::x86::add_int64(&A->GetShape(), &B->GetShape(), &C->GetShape(), A->GetBufferPtr<const int64_t>(),
                                      B->GetBufferPtr<const int64_t>(), C->GetBufferPtr<int64_t>());
    } else {
        LOG(ERROR) << "unsupported datatype " << common::GetDataTypeStr(data_type);
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
