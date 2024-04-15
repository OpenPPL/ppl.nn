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

#include "multi_head_cache_attention_kernel.h"

#include "ppl/common/destructor.h"

#include "ppl/kernel/llm/cuda/pmx/multi_head_cache_attention.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {


ppl::common::RetCode DynamicBatchingMultiHeadCacheAttentionKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(query, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(current_key, 1);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(current_value, 2);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(seqstarts, 3);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(kvstarts, 4);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(cachestarts, 5);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(start_pos, 6);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(decoding_batches, 7);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(max_seqlen, 8);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(max_kvlen, 9);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(cache, 10);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(scale, 11);
    PPLNN_LLM_CUDA_OPTIONAL_INPUT(attn_mask, 12);

    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(attn_output, 0);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [query]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(query);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [current_key]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(current_key);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [current_value]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(current_value);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [seqstarts]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(seqstarts);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [kvstarts]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(kvstarts);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [cachestarts]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(cachestarts);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [start_pos]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(start_pos);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [decoding_batches]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(decoding_batches);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [max_seqlen]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(max_seqlen);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [max_kvlen]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(max_kvlen);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [cache]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(cache);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [scale]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(scale);
    if (scale) {
        PPLNN_LLM_CUDA_DEBUG_TRACE("Input [scale]:\n");
        PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(scale);
    }
    if (attn_mask) {
        PPLNN_LLM_CUDA_DEBUG_TRACE("Input [attn_mask]:\n");
        PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(attn_mask);
    }

    PPLNN_LLM_CUDA_DEBUG_TRACE("num_heads: %d\n", param_->num_heads);
    PPLNN_LLM_CUDA_DEBUG_TRACE("num_kv_heads: %d\n", param_->num_kv_heads);
    PPLNN_LLM_CUDA_DEBUG_TRACE("head_dim: %d\n", param_->head_dim);
    PPLNN_LLM_CUDA_DEBUG_TRACE("is_causal: %d\n", param_->is_causal);
    PPLNN_LLM_CUDA_DEBUG_TRACE("num_layer: %d\n", param_->num_layer);
    PPLNN_LLM_CUDA_DEBUG_TRACE("layer_idx: %d\n", param_->layer_idx);
    PPLNN_LLM_CUDA_DEBUG_TRACE("quant_bit: %d\n", param_->quant_bit);
    PPLNN_LLM_CUDA_DEBUG_TRACE("quant_group: %d\n", param_->quant_group);
    PPLNN_LLM_CUDA_DEBUG_TRACE("cache_mode: %d\n", param_->cache_mode);
    PPLNN_LLM_CUDA_DEBUG_TRACE("cache_layout: %d\n", param_->cache_layout);

    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();


    if (param_->quant_bit != 8 && param_->quant_group != 8) {
        LOG(ERROR) << "currently only support quant_bit == quant_group == 8";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (param_->cache_mode != 0) {
        LOG(ERROR) << "currently only support cache_mode == 0";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (!scale || scale->GetShape()->CalcElementsExcludingPadding() == 0) {
        LOG(ERROR) << "currently only support qunatized cache but scale not found";
        return ppl::common::RC_UNSUPPORTED;
    }

    TensorShape* attn_mask_shape = nullptr;
    void* attn_mask_data = nullptr;
    if (attn_mask && attn_mask->GetShape()->CalcElementsExcludingPadding() > 0) {
        attn_mask_shape = attn_mask->GetShape();
        attn_mask_data = attn_mask->GetBufferPtr();
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

    const int64_t batch = start_pos->GetShape()->GetDim(0);

    int64_t cache_stride_s = 0;
    int64_t cache_stride_l = 0;
    int64_t cache_stride_h = 0;
    int64_t cache_stride_kv = 0;

    if (param_->cache_layout == 0) {
        const int64_t max_tokens = cache->GetShape()->GetDim(0);
        (void)max_tokens;
        // (MaxT,L,2,H,Dh)
        cache_stride_s = param_->num_layer * 2 * param_->num_kv_heads * param_->head_dim;
        cache_stride_l = 2 * param_->num_kv_heads * param_->head_dim;
        cache_stride_h = param_->head_dim;
        cache_stride_kv = param_->num_kv_heads * param_->head_dim;
    } else if (param_->cache_layout == 3) {
        const int64_t max_tokens = cache->GetShape()->GetDim(3);
        // (L,2,H,MaxT,Dh)
        cache_stride_s = param_->head_dim;
        cache_stride_l = 2 * param_->num_kv_heads * max_tokens * param_->head_dim;
        cache_stride_h = max_tokens * param_->head_dim;
        cache_stride_kv = param_->num_kv_heads * max_tokens * param_->head_dim;
    } else {
        LOG(ERROR) << "currently only support cache_layout == 0 or cache_layout == 3";
        return ppl::common::RC_UNSUPPORTED;
    }

    auto p_ret = ppl::kernel::llm::cuda::pmx::dynamic_batching_multi_head_cache_attention_prepare(
        GetStream(),
        GetCudaDevice()->GetDeviceProp(),
        query->GetShape(),
        query->GetBufferPtr(),
        current_key->GetShape(),
        current_key->GetBufferPtr(),
        current_value->GetShape(),
        current_value->GetBufferPtr(),
        attn_mask_shape,
        attn_mask_data,
        seqstarts->GetBufferPtr(),
        kvstarts->GetBufferPtr(),
        cachestarts->GetBufferPtr(),
        start_pos->GetBufferPtr(),
        param_->is_causal,
        batch,
        decodeing_batches_val,
        max_seqlen_val,
        max_kvlen_val,
        param_->layer_idx,
        param_->num_layer,
        param_->num_heads,
        param_->num_kv_heads,
        param_->head_dim,
        param_->cache_mode,
        cache_stride_s,
        cache_stride_l,
        cache_stride_h,
        cache_stride_kv,
        cache->GetBufferPtr(),
        scale->GetBufferPtr(),
        attn_output->GetShape(),
        attn_output->GetBufferPtr()
    );

    if (p_ret.first != ppl::common::RC_SUCCESS) {
        return p_ret.first;
    }

    auto &cfg = p_ret.second;

    BufferDesc tmpbuffer_desc;
    auto status = GetCudaDevice()->AllocTmpBuffer(cfg.temp_buffer_size, &tmpbuffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << cfg.temp_buffer_size << "] for kernel[" << GetName()
                << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    ppl::common::Destructor multi_block_tmpbuffer_guard([this, &tmpbuffer_desc]() -> void {
        GetCudaDevice()->FreeTmpBuffer(&tmpbuffer_desc);
    });
    cfg.temp_buffer = tmpbuffer_desc.addr;

    return ppl::kernel::llm::cuda::pmx::dynamic_batching_multi_head_cache_attention(
        GetStream(),
        cfg
    );
}

}}}}} // namespace ppl::nn::llm::cuda::pmx
