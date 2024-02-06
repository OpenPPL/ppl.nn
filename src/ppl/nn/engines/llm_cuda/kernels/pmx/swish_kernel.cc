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

#include "swish_kernel.h"

#include "ppl/kernel/llm/cuda/pmx/silu.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode SwishKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(input, 0);
    PPLNN_LLM_CUDA_OPTIONAL_INPUT(gate, 1);
    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(output, 0);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [input]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(input);

    PPLNN_LLM_CUDA_DEBUG_TRACE("beta: %f\n", param_->beta);

    void* gate_data = nullptr;
    if (gate) {
        PPLNN_LLM_CUDA_DEBUG_TRACE("Input [gate]:\n");
        PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(gate);
        gate_data = gate->GetBufferPtr();
    }

    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();

    bool can_trans_input = ctx->IsLastConsumerOfInput(0) && input->GetType() == TENSORTYPE_NORMAL;
    bool can_trans_gate = gate && ctx->IsLastConsumerOfInput(1) && gate->GetType() == TENSORTYPE_NORMAL;

    auto input_data = input->GetBufferPtr();
    if (can_trans_input) {
        output->TransferBufferFrom(input);
    } else if (can_trans_gate) {
        output->TransferBufferFrom(gate);
    } else {
        PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(output);
    }
    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(output);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [output]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(output);

    return ppl::kernel::llm::cuda::pmx::silu(
        GetStream(), 
        input->GetShape(),
        input->GetBufferPtr(),
        gate_data,
        param_->beta,
        output->GetBufferPtr());
}


}}}}} // namespace ppl::nn::llm::cuda::pmx
