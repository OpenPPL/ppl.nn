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

#include "parallel_embedding_kernel.h"

#include "ppl/common/cuda/nccl_utils.h"
#include "ppl/common/destructor.h"

#include "ppl/kernel/llm/cuda/pmx/parallel_embedding.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode ParallelEmbeddingKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(indices, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(weight, 1);
    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(output, 0);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [indices]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(indices);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [weight]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(weight);

    PPLNN_LLM_CUDA_DEBUG_TRACE("num_embeddings: %d\n", param_->num_embeddings);
    PPLNN_LLM_CUDA_DEBUG_TRACE("embedding_dims: %d\n", param_->embedding_dims);
    PPLNN_LLM_CUDA_DEBUG_TRACE("max_norm: %f\n", param_->max_norm);
    PPLNN_LLM_CUDA_DEBUG_TRACE("norm_type: %f\n", param_->norm_type);
    PPLNN_LLM_CUDA_DEBUG_TRACE("padding_idx: %d\n", param_->padding_idx);

    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(output);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [output]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(output);

    auto indices_shape = indices->GetShape();
    auto weight_shape = weight->GetShape();
    auto output_shape = output->GetShape();

    if (ppl::common::DATATYPE_FLOAT16 != weight_shape->GetDataType()) {
        LOG(ERROR) << "currently only support fp16";
        return ppl::common::RC_UNSUPPORTED;
    }

    auto nccl_param = GetTensorParallelNcclParam();

    uint64_t gather_buffer_size = 0;
    void *gather_buffer = nullptr;
    if (nccl_param->size > 1) {
        gather_buffer_size = output_shape->CalcBytesExcludingPadding();
    }

    BufferDesc tmp_buffer_desc;
    auto status = GetCudaDevice()->AllocTmpBuffer(gather_buffer_size, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << gather_buffer_size << "] for kernel[" << GetName()
                << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    ppl::common::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
        GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc);
    });
    gather_buffer = tmp_buffer_desc.addr;

    return ppl::kernel::llm::cuda::pmx::parallel_embedding(
        GetStream(),
        indices_shape,
        indices->GetBufferPtr(),
        weight_shape,
        weight->GetBufferPtr(),
        param_->num_embeddings,
        param_->embedding_dims,
        param_->max_norm,
        param_->norm_type,
        param_->padding_idx,
        nccl_param,
        gather_buffer,
        output_shape,
        output->GetBufferPtr()
    );

}

}}}}} // namespace ppl::nn::llm::cuda::pmx
