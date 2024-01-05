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

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {


ppl::common::RetCode DynamicBatchingMultiHeadAttentionKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(query, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(key, 1);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(value, 2);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(seqstarts, 3);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(kvstarts, 4);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(decoding_batches, 5);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(max_seqlen, 6);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(max_kvlen, 7);
    PPLNN_LLM_CUDA_OPTIONAL_INPUT(attn_mask, 8);

    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(attn_output, 0);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [query]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(query);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [key]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(key);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [value]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(value);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [seqstarts]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(seqstarts);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [kvstarts]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(kvstarts);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [decoding_batches]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(decoding_batches);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [max_seqlen]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(max_seqlen);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [max_kvlen]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(max_kvlen);
    if (attn_mask) {
        PPLNN_LLM_CUDA_DEBUG_TRACE("Input [attn_mask]:\n");
        PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(attn_mask);
    }

    PPLNN_LLM_CUDA_DEBUG_TRACE("num_heads: %d\n", param_->num_heads);
    PPLNN_LLM_CUDA_DEBUG_TRACE("num_kv_heads: %d\n", param_->num_kv_heads);
    PPLNN_LLM_CUDA_DEBUG_TRACE("head_dim: %d\n", param_->head_dim);
    PPLNN_LLM_CUDA_DEBUG_TRACE("is_causal: %d\n", param_->is_causal);

    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();

    if (param_->is_causal == false) {
        LOG(ERROR) << "currently only support is_causal == true";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (param_->num_heads != param_->num_kv_heads) {
        LOG(ERROR) << "currently not support group query attention";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (attn_mask && attn_mask->GetShape()->CalcElementsExcludingPadding() > 0) {
        LOG(ERROR) << "currently do not support attn_mask";
        return ppl::common::RC_UNSUPPORTED;
    }

    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(attn_output);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [attn_output]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(attn_output);

    int64_t decodeing_batches_val = 0;
    if (ppl::common::RC_SUCCESS != decoding_batches->CopyToHost(&decodeing_batches_val)) {
        LOG(ERROR) << "decoding_batches->CopyToHost() failed";
        return ppl::common::RC_DEVICE_MEMORY_ERROR;
    }

    int64_t max_seqlen_val = 0;
    if (ppl::common::RC_SUCCESS != max_seqlen->CopyToHost(&max_seqlen_val)) {
        LOG(ERROR) << "max_seqlen->CopyToHost() failed";
        return ppl::common::RC_DEVICE_MEMORY_ERROR;
    }

    int64_t max_kvlen_val = 0;
    if (ppl::common::RC_SUCCESS != max_kvlen->CopyToHost(&max_kvlen_val)) {
        LOG(ERROR) << "max_kvlen->CopyToHost() failed";
        return ppl::common::RC_DEVICE_MEMORY_ERROR;
    }

    LOG(ERROR) << "currently do not support this op";
    return ppl::common::RC_UNSUPPORTED;
}

}}}}} // namespace ppl::nn::llm::cuda::pmx
