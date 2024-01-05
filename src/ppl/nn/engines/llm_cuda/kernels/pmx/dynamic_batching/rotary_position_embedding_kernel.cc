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

#include "rotary_position_embedding_kernel.h"

#include "ppl/kernel/llm/cuda/pmx/rotary_position_embedding.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode DynamicBatchingRotaryPositionEmbeddingKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(query, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(key, 1);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(seqstarts, 2);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(start_pos, 3);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(max_seqlen, 4);

    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(rotated_query, 0);
    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(rotated_key, 1);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [query]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(query);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [key]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(key);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [seqstarts]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(seqstarts);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [start_pos]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(start_pos);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [max_seqlen]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(max_seqlen);

    PPLNN_LLM_CUDA_DEBUG_TRACE("theta: %f\n", param_->theta);
    PPLNN_LLM_CUDA_DEBUG_TRACE("rotary_dim: %d\n", param_->rotary_dim);
    PPLNN_LLM_CUDA_DEBUG_TRACE("bypass_key: %d\n", param_->bypass_key);

    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();

    bool can_trans_query = ctx->IsLastConsumerOfInput(0) && query->GetType() == TENSORTYPE_NORMAL;
    bool can_trans_key = ctx->IsLastConsumerOfInput(1) && key->GetType() == TENSORTYPE_NORMAL;

    auto query_data = query->GetBufferPtr();
    auto key_data = key->GetBufferPtr();

    if (can_trans_query) {
        rotated_query->TransferBufferFrom(query);
    } else {
        PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(rotated_query);
    }
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [rotated_query]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(rotated_query);

    if (can_trans_key) {
        rotated_key->TransferBufferFrom(key);
    } else {
        PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(rotated_key);
    }
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [rotated_key]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(rotated_key);

    auto query_shape = query->GetShape();
    auto key_shape = key->GetShape();

    if (ppl::common::DATATYPE_FLOAT16 != query_shape->GetDataType()) {
        LOG(ERROR) << "currently only support fp16";
        return ppl::common::RC_UNSUPPORTED;
    }

    int64_t max_seqlen_val = 0;
    if (ppl::common::RC_SUCCESS != max_seqlen->CopyToHost(&max_seqlen_val)) {
        LOG(ERROR) << "max_seqlen->CopyToHost() failed";
        return ppl::common::RC_DEVICE_MEMORY_ERROR;
    }

    int64_t batch = start_pos->GetShape()->GetDim(0);
    int64_t head_dim = query_shape->GetDim(2);
    int64_t rotary_dim = param_->rotary_dim == 0 ? head_dim : param_->rotary_dim;
    int64_t num_heads = query_shape->GetDim(1);
    int64_t num_key_heads = key_shape->GetDim(1);

    return ppl::kernel::llm::cuda::pmx::dynamic_batching_rotary_position_embedding(
        GetStream(),
        query_shape,
        query_data,
        key_shape,
        key_data,
        start_pos->GetBufferPtr(),
        seqstarts->GetBufferPtr(),
        param_->theta,
        param_->bypass_key,
        rotary_dim,
        batch,
        num_heads,
        num_key_heads,
        max_seqlen_val,
        rotated_query->GetShape(),
        rotated_query->GetBufferPtr(),
        rotated_key->GetShape(),
        rotated_key->GetBufferPtr()
    );
}

}}}}} // namespace ppl::nn::llm::cuda::pmx
