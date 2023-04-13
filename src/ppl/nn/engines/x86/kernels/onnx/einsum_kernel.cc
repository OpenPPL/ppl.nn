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

#include "ppl/nn/engines/x86/kernels/onnx/einsum_kernel.h"
#include "ppl/kernel/x86/fp32/matmul.h"
#include "ppl/kernel/x86/fp32/gemm.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode EinSumKernel::DoExecute(KernelExecContext* ctx) {
    std::vector<TensorImpl*> inputs(ctx->GetInputCount());
    PPLNN_X86_REQUIRED_OUTPUT(output, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());

    for (uint32_t i = 0; i < ctx->GetInputCount(); ++i) {
        inputs[i] = ctx->GetInput<TensorImpl>(i);
        PPLNN_X86_DEBUG_TRACE("Input [inputs[%u]]:\n", i);
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(inputs[i]);
    }


    PPLNN_X86_DEBUG_TRACE("equation: %s\n", param_->equation.c_str());
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    PPLNN_X86_REALLOC_TENSOR_BUFFER(output);
    PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);

    const auto data_type = output->GetShape()->GetDataType();

    std::string sim_equ;
    sim_equ.reserve(param_->equation.size());
    for (auto c : param_->equation)
        if (c != ' ') sim_equ.push_back(c);

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        if (sim_equ == "i,j->ij") {
            return ppl::kernel::x86::gemm_fp32(GetISA(),
                inputs[0]->GetBufferPtr<float>(), inputs[1]->GetBufferPtr<float>(), nullptr, nullptr,
                ppl::kernel::x86::gemm_m_type::NOTRANS, ppl::kernel::x86::gemm_m_type::NOTRANS,
                ppl::kernel::x86::gemm_m_type::EMPTY, ppl::kernel::x86::gemm_m_type::EMPTY,
                inputs[0]->GetShape()->GetDim(0), inputs[1]->GetShape()->GetDim(0), 1,
                1, inputs[1]->GetShape()->GetDim(0), inputs[1]->GetShape()->GetDim(0), 0,
                1.0, 0.0, 0.0, 0.0, ppl::kernel::x86::gemm_post::NONE, output->GetBufferPtr<float>());
        }
        if (sim_equ == "bij,bjd->bid") {
            return ppl::kernel::x86::matmul_ndarray_fp32(GetISA(),
                inputs[0]->GetShape(), inputs[1]->GetShape(), output->GetShape(),
                inputs[0]->GetBufferPtr<float>(), inputs[1]->GetBufferPtr<float>(),
                ppl::kernel::x86::gemm_m_type::NOTRANS, ppl::kernel::x86::gemm_m_type::NOTRANS,
                false, output->GetBufferPtr<float>());
        }
        if (sim_equ == "bid,bjd->bij") {
            return ppl::kernel::x86::matmul_ndarray_fp32(GetISA(),
                inputs[0]->GetShape(), inputs[1]->GetShape(), output->GetShape(),
                inputs[0]->GetBufferPtr<float>(), inputs[1]->GetBufferPtr<float>(),
                ppl::kernel::x86::gemm_m_type::NOTRANS, ppl::kernel::x86::gemm_m_type::TRANS,
                false, output->GetBufferPtr<float>());
        }
        LOG(ERROR) << "unsupported equation: " << param_->equation;
        return ppl::common::RC_UNSUPPORTED;
    } else {
        LOG(ERROR) << "unsupported datatype: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
