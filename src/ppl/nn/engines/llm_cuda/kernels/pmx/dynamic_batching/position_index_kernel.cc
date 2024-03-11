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

#include "position_index_kernel.h"

#include "ppl/kernel/llm/cuda/pmx/position_index.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode DynamicBatchingPositionIndexKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(sequence, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(seqstarts, 1);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(start_pos, 2);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(max_seqlen, 3);

    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(position_idx, 0);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [sequence]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(sequence);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [seqstarts]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(seqstarts);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [start_pos]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(start_pos);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [max_seqlen]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(max_seqlen);

    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();

    bool can_trans = ctx->IsLastConsumerOfInput(0) && sequence->GetType() == TENSORTYPE_NORMAL;
    if (can_trans) {
        position_idx->TransferBufferFrom(sequence);
    } else {
        PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(position_idx);
    }
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [position_idx]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(position_idx);

    int64_t max_seqlen_val = 0;
    if (ppl::common::RC_SUCCESS != max_seqlen->CopyToHost(&max_seqlen_val)) {
        LOG(ERROR) << "max_seqlen->CopyToHost() failed";
        return ppl::common::RC_DEVICE_MEMORY_ERROR;
    }

    int64_t batch = start_pos->GetShape()->GetDim(0);

    return ppl::kernel::llm::cuda::pmx::dynamic_batching_position_index(
        GetStream(),
        start_pos->GetBufferPtr(),
        seqstarts->GetBufferPtr(),
        batch,
        max_seqlen_val,
        position_idx->GetBufferPtr()
    );
}

}}}}} // namespace ppl::nn::llm::cuda::pmx
