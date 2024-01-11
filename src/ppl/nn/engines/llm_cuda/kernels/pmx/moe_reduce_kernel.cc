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

#include "moe_reduce_kernel.h"

#include "ppl/kernel/llm/cuda/pmx/moe_reduce.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode MoeReduceKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(y_permute_expand, 0);
    PPLNN_LLM_CUDA_OPTIONAL_INPUT(expert_weights, 1);
    PPLNN_LLM_CUDA_OPTIONAL_INPUT(invert_permutation, 2);

    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(y_reduced, 0);

    // trace info
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [y_permute_expand]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(y_permute_expand);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [expert_weights]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(expert_weights);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [invert_permutation]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(invert_permutation);

    PPLNN_LLM_CUDA_DEBUG_TRACE("num_experts_per_token: %d\n", param_->num_experts_per_token);

    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();

    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(y_reduced);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [y_reduced]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(y_reduced);

    return ppl::kernel::llm::cuda::pmx::moe_reduce(
        GetStream(), y_permute_expand->GetShape(), y_permute_expand->GetBufferPtr(), expert_weights->GetBufferPtr(),
        invert_permutation->GetBufferPtr(), param_->num_experts_per_token, y_reduced->GetBufferPtr());
}

}}}}} // namespace ppl::nn::llm::cuda::pmx