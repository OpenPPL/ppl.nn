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

#include "multi_head_attention_kernel.h"
#include "ppl/kernel/llm/cuda/pmx/multi_head_attention.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {


ppl::common::RetCode MultiHeadAttentionKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(query, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(key, 1);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(value, 2);
    PPLNN_LLM_CUDA_OPTIONAL_INPUT(attn_mask, 3);

    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(attn_output, 0);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [query]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(query);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [key]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(key);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [value]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(value);
    if (attn_mask) {
        PPLNN_LLM_CUDA_DEBUG_TRACE("Input [attn_mask]:\n");
        PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(attn_mask);
    }

    PPLNN_LLM_CUDA_DEBUG_TRACE("num_heads: %d\n", param_->num_heads);
    PPLNN_LLM_CUDA_DEBUG_TRACE("num_kv_heads: %d\n", param_->num_kv_heads);
    PPLNN_LLM_CUDA_DEBUG_TRACE("head_dim: %d\n", param_->head_dim);
    PPLNN_LLM_CUDA_DEBUG_TRACE("is_causal: %d\n", param_->is_causal);

    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();

    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(attn_output);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [attn_output]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(attn_output);

    void *attn_mask_data = nullptr;
    TensorShape *attn_mask_shape = nullptr;

    if (attn_mask && attn_mask->GetShape()->CalcElementsExcludingPadding() > 0) {
        attn_mask_data = attn_mask->GetBufferPtr();
        attn_mask_shape = attn_mask->GetShape();
    }

    return ppl::kernel::llm::cuda::pmx::multi_head_attention(
        GetStream(),
        GetCudaDevice()->GetDeviceProp(),
        query->GetShape(),
        query->GetBufferPtr(),
        key->GetShape(),
        key->GetBufferPtr(),
        value->GetShape(),
        value->GetBufferPtr(),
        attn_mask_shape,
        attn_mask_data,
        param_->is_causal,
        param_->num_heads,
        param_->num_kv_heads,
        param_->head_dim,
        attn_output->GetShape(),
        attn_output->GetBufferPtr());
}

}}}}} // namespace ppl::nn::llm::cuda::pmx
