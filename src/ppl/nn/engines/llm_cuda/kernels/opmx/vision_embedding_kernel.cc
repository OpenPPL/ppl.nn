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

#include "vision_embedding_kernel.h"

#include "ppl/common/destructor.h"

#include "ppl/kernel/llm/cuda/pmx/vision_embedding.h"

#ifdef PPLNN_CUDA_ENABLE_CUDNN
#include <cudnn.h>
#endif

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace opmx {

ppl::common::RetCode VisionEmbeddingKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

#ifdef PPLNN_CUDA_ENABLE_CUDNN
    PPLNN_LLM_CUDA_REQUIRED_INPUT(pixel_values, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(class_weight, 1);
    // TODO: fix optional input to required input. refer to pmx doc
    PPLNN_LLM_CUDA_REQUIRED_INPUT(patch_weight, 2);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(position_weight, 3);
    PPLNN_LLM_CUDA_OPTIONAL_INPUT(patch_bias, 4);
    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(output_embeddings, 0);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [pixel_values]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(pixel_values);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [class_weight]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(class_weight);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [patch_weight]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(patch_weight);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [position_weight]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(position_weight);
    if (patch_bias) {
        PPLNN_LLM_CUDA_DEBUG_TRACE("Input [patch_bias]:\n");
        PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(patch_bias);
    }

    PPLNN_LLM_CUDA_DEBUG_TRACE("hidden_dim: %d\n", param_->hidden_dim);
    PPLNN_LLM_CUDA_DEBUG_TRACE("patch_size: %d\n", param_->patch_size);

    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();

    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(output_embeddings);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [output_embeddings]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(output_embeddings);

    ppl::kernel::llm::cuda::pmx::vision_embedding_config config;

    cudnnStatus_t cudnn_status;
    config.cudnn_handle = GetCudaDevice()->GetCudnnHandle();
    if (config.cudnn_handle == nullptr) {
        LOG(ERROR) << "cudnn handle is invalid.";
        return ppl::common::RC_OTHER_ERROR;
    }

    config.bias_term = patch_bias != nullptr;
    config.hidden_dim = param_->hidden_dim;
    config.patch_size = param_->patch_size;
    auto image_shape = pixel_values->GetShape();
    config.batch_size    = image_shape->GetDim(0);
    config.image_channel = image_shape->GetDim(1);
    config.image_size    = image_shape->GetDim(2);
    if (image_shape->GetDim(2) != image_shape->GetDim(3)) {
        LOG(ERROR) << "an image with unequal height and width isn't supported.";
        return ppl::common::RC_UNSUPPORTED;
    }

    // TODO: check retcode here
    auto pplcommon_status = ppl::kernel::llm::cuda::pmx::vision_embedding_preprocessing(config);
    if (pplcommon_status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "ppl::kernel::llm::cuda::pmx::vision_embedding_preprocessing() in kernel "
                   << GetName() << "] failed: " << ppl::common::GetRetCodeStr(pplcommon_status);
        return pplcommon_status;
    }

    BufferDesc buffers_desc;
    pplcommon_status = GetCudaDevice()->AllocTmpBuffer(config.total_buffer_size, &buffers_desc);
    if (pplcommon_status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc buffers size[" << config.total_buffer_size << "] for kernel["
                   << GetName() << "] failed: " << ppl::common::GetRetCodeStr(pplcommon_status);
        return pplcommon_status;
    }
    ppl::common::Destructor buffer_guard([this, &buffers_desc]() -> void {
        GetCudaDevice()->FreeTmpBuffer(&buffers_desc);
    });
    config.buffer_addr = buffers_desc.addr;

    // TODO: check retcode here
    pplcommon_status = ppl::kernel::llm::cuda::pmx::vision_embedding(
        GetStream(),
        config,
        pixel_values->GetBufferPtr(),
        class_weight->GetBufferPtr(),    // [hidden_dim]
        patch_weight->GetBufferPtr(),    // [hidden_dim, image_channel, patch_size, patch_size]
        position_weight->GetBufferPtr(), // [num_positions * hidden_dim]
        patch_bias->GetBufferPtr(),      // [hidden_dim]
        output_embeddings->GetBufferPtr()
    );
    if (pplcommon_status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "ppl::kernel::llm::cuda::pmx::vision_embedding() in kernel "
                   << GetName() << "] failed: " << ppl::common::GetRetCodeStr(pplcommon_status);
        return pplcommon_status;
    }

    // TODO: check retcode here
    pplcommon_status =  ppl::kernel::llm::cuda::pmx::vision_embedding_postprocessing(config);
    if (pplcommon_status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "ppl::kernel::llm::cuda::pmx::vision_embedding_postprocessing() in kernel "
                   << GetName() << "] failed: " << ppl::common::GetRetCodeStr(pplcommon_status);
        return pplcommon_status;
    }

    return ppl::common::RC_SUCCESS;
#else
    LOG(ERROR) << "VisionEmbedding need cudnn, please recompile with PPLNN_CUDA_ENABLE_CUDNN=ON";
    return ppl::common::RC_UNSUPPORTED;
#endif
}

}}}}} // namespace ppl::nn::llm::cuda::opmx
