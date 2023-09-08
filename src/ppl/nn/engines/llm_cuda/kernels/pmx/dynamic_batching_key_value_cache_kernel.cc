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

#include "dynamic_batching_key_value_cache_kernel.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {


ppl::common::RetCode DynamicBatchingKeyValueCacheKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(current_key, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(current_value, 1);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(seqstarts, 2);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(kvstarts, 3);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(cachestarts, 4);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(start_pos, 5);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(max_seqlen, 6);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(max_kvlen, 7);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(cache, 8);
    PPLNN_LLM_CUDA_OPTIONAL_INPUT(scale, 9);

    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(key, 0);
    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(value, 1);

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
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [max_seqlen]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(max_seqlen);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [max_kvlen]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(max_kvlen);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [cache]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(cache);
    if (scale) {
        PPLNN_LLM_CUDA_DEBUG_TRACE("Input [scale]:\n");
        PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(scale);
    }

    PPLNN_LLM_CUDA_DEBUG_TRACE("num_layer: %d\n", param_->num_layer);
    PPLNN_LLM_CUDA_DEBUG_TRACE("layer_idx: %d\n", param_->layer_idx);
    PPLNN_LLM_CUDA_DEBUG_TRACE("quant_bit: %d\n", param_->quant_bit);
    PPLNN_LLM_CUDA_DEBUG_TRACE("quant_group: %d\n", param_->quant_group);
    PPLNN_LLM_CUDA_DEBUG_TRACE("num_repeat: %d\n", param_->num_repeat);
    PPLNN_LLM_CUDA_DEBUG_TRACE("cache_mode: %d\n", param_->cache_mode);
    PPLNN_LLM_CUDA_DEBUG_TRACE("cache_layout: %d\n", param_->cache_layout);

    if (!scale) {
        LOG(ERROR) << "currently only support qunatized cache but scale not found";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (param_->quant_bit != 8 && param_->quant_group != 8) {
        LOG(ERROR) << "currently only support quant_bit == quant_group == 8";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (param_->cache_layout != 0) {
        LOG(ERROR) << "currently only support cache_layout == 0";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (param_->cache_mode != 0) {
        LOG(ERROR) << "currently only support cache_mode == 0";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (param_->num_repeat != 1) {
        LOG(ERROR) << "currently only support num_repeat == 1";
        return ppl::common::RC_UNSUPPORTED;
    }

    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(key);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [key]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(key);
    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(value);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [value]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(value);

    auto current_key_shape = current_key->GetShape();

    if (ppl::common::DATATYPE_FLOAT16 != current_key_shape->GetDataType()) {
        LOG(ERROR) << "currently only support fp16";
        return ppl::common::RC_UNSUPPORTED;
    }

    LOG(ERROR) << "currently do not support this op";
    return ppl::common::RC_UNSUPPORTED;
}

}}}}} // namespace ppl::nn::llm::cuda::pmx
