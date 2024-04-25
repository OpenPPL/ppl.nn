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

#include "ppl/common/cuda/nccl_utils.h"
#include "ppl/common/destructor.h"

#include "ppl/kernel/llm/cuda/pmx/vision_embedding.h"

#include <cudnn.h>

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode VisionEmbeddingKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(pixel_values, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(cls_emb_weight, 1);
    PPLNN_LLM_CUDA_OPTIONAL_INPUT(patch_emb_weight, 2);
    PPLNN_LLM_CUDA_OPTIONAL_INPUT(pos_emb_weight, 3);
    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(vision_embeddings, 0);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [pixel_values]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(pixel_values);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [cls_emb_weight]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(cls_emb_weight);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [patch_emb_weight]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(patch_emb_weight);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [pos_emb_weight]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(pos_emb_weight);

    PPLNN_LLM_CUDA_DEBUG_TRACE("hidden_dim: %d\n", param_->hidden_dim);
    PPLNN_LLM_CUDA_DEBUG_TRACE("image_size: %d\n", param_->image_size);
    PPLNN_LLM_CUDA_DEBUG_TRACE("patch_size: %d\n", param_->patch_size);

    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();

    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(vision_embeddings);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [vision_embeddings]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(vision_embeddings);

    cudnnStatus_t status0;
    cudnnHandle_t cudnn_handle;
    if (GetCudaDevice()->GetCudnnHandle() != nullptr) {
        cudnn_handle = GetCudaDevice()->GetCudnnHandle();
    }
    else {
        status0 = cudnnCreate(&cudnn_handle);
        if (status0 != CUDNN_STATUS_SUCCESS) {
            LOG(ERROR) << "failed to create a cudnn handle with error: " << cudnnGetErrorString(status0);
            return ppl::common::RC_OTHER_ERROR;
        }
        GetCudaDevice()->SetCudnnHandle(cudnn_handle);
    }

    const int32_t hidden_dim = param_->hidden_dim;
    const int32_t image_size = param_->image_size;
    const int32_t patch_size = param_->patch_size;
    auto image_shape = pixel_values->GetShape();
    const int32_t batch_size = image_shape->GetDim(0);
    const int32_t image_channel = image_shape->GetDim(1);
    const int32_t image_size1 = image_shape->GetDim(2);
    if (image_size != image_size1) {
        LOG(ERROR) << "The image size is inconsistant with the image data.";
        return ppl::common::RC_INVALID_VALUE;
    }
    const int32_t grid = image_size / patch_size;
    cudnnTensorDescriptor_t image_desc, patch_desc0, patch_desc1;
    status0 = cudnnCreateTensorDescriptor(&image_desc);
    if (status0 != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to create the tensor descriptor of image with error: " << cudnnGetErrorString(status0);
        return ppl::common::RC_OTHER_ERROR;
    }
    status0 = cudnnCreateTensorDescriptor(&patch_desc0);
    if (status0 != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to create the tensor descriptor of convolution output with error: " << cudnnGetErrorString(status0);
        return ppl::common::RC_OTHER_ERROR;
    }
    status0 = cudnnCreateTensorDescriptor(&patch_desc1);
    if (status0 != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to create the tensor descriptor of transposed convolution output with error: " << cudnnGetErrorString(status0);
        return ppl::common::RC_OTHER_ERROR;
    }
    status0 = cudnnSetTensor4dDescriptor(image_desc, CUDNN_TENSOR_NCHW,
             CUDNN_DATA_HALF, batch_size, image_channel, image_size, image_size);
    if (status0 != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set the tensor descriptor of image with error: " << cudnnGetErrorString(status0);
        return ppl::common::RC_OTHER_ERROR;
    }
    status0 = cudnnSetTensor4dDescriptor(patch_desc0, CUDNN_TENSOR_NCHW,
             CUDNN_DATA_HALF, batch_size, hidden_dim, grid, grid);
    if (status0 != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set the tensor descriptor of convolution output with error: " << cudnnGetErrorString(status0);
        return ppl::common::RC_OTHER_ERROR;
    }
    status0 = cudnnSetTensor4dDescriptor(patch_desc1, CUDNN_TENSOR_NHWC,
             CUDNN_DATA_HALF, batch_size, hidden_dim, grid, grid);
    if (status0 != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set the tensor descriptor of transposed convolution output with error: " << cudnnGetErrorString(status0);
        return ppl::common::RC_OTHER_ERROR;
    }

    cudnnFilterDescriptor_t filter_desc;
    status0 = cudnnCreateFilterDescriptor(&filter_desc);
    if (status0 != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to create the filter descriptor with error: " << cudnnGetErrorString(status0);
        return ppl::common::RC_OTHER_ERROR;
    }
    status0 = cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_HALF,
             CUDNN_TENSOR_NCHW, hidden_dim, image_channel, patch_size, patch_size);
    if (status0 != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set the filter descriptor with error: " << cudnnGetErrorString(status0);
        return ppl::common::RC_OTHER_ERROR;
    }

    cudnnConvolutionDescriptor_t conv_desc;
    status0 = cudnnCreateConvolutionDescriptor(&conv_desc);
    if (status0 != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to create the convolution descriptor with error: " << cudnnGetErrorString(status0);
        return ppl::common::RC_OTHER_ERROR;
    }
    status0 = cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, patch_size, patch_size, 1, 1,
             CUDNN_CONVOLUTION, CUDNN_DATA_HALF);
    if (status0 != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set the convolution descriptor with error: " << cudnnGetErrorString(status0);
        return ppl::common::RC_OTHER_ERROR;
    }
    status0 = cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);
    if (status0 != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set the convolution math type with error: " << cudnnGetErrorString(status0);
        return ppl::common::RC_OTHER_ERROR;
    }

    size_t workspace_size;
    status0 = cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, image_desc,
             filter_desc, conv_desc, patch_desc0,
             CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
             &workspace_size);
    if (status0 != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to get the workspace size of convolution with error: " << cudnnGetErrorString(status0);
        return ppl::common::RC_OTHER_ERROR;
    }
    size_t size = batch_size * hidden_dim * grid * grid * sizeof(half);

    workspace_size = ((workspace_size + sizeof(half)) >> 1) << 1;
    size_t total_size = workspace_size + size * 2;
    BufferDesc buffers_desc;
    auto status1 = GetCudaDevice()->AllocTmpBuffer(total_size, &buffers_desc);
    if (status1 != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc buffers size[" << total_size << "] for kernel["
                   << GetName() << "] failed: " << ppl::common::GetRetCodeStr(status1);
        return status1;
    }
    ppl::common::Destructor patch1_guard([this, &buffers_desc]() -> void {
        GetCudaDevice()->FreeTmpBuffer(&buffers_desc);
    });
    void* workspace = buffers_desc.addr;
    void* patch_embeds0 = workspace + workspace_size;
    void* patch_embeds1 = patch_embeds0 + size;

    ppl::kernel::llm::cuda::pmx::vision_embedding(
        GetStream(),
        cudnn_handle,
        image_desc,
        pixel_values->GetBufferPtr(),
        filter_desc,
        patch_emb_weight->GetBufferPtr(),  // weight of convolution filter
        conv_desc,
        workspace,
        workspace_size,
        patch_desc0,
        patch_embeds0,  // output of cudnnConvolutionForward()
        patch_desc1,
        patch_embeds1,  // output of cudnnTransformTensor()
        cls_emb_weight->GetBufferPtr(),  // [hidden_dim]
        pos_emb_weight->GetBufferPtr(),  // [num_positions * hidden_dim]
        grid,
        batch_size,
        hidden_dim,
        vision_embeddings->GetBufferPtr()
    );

    status0 = cudnnDestroyTensorDescriptor(image_desc);
    if (status0 != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to destroy the cudnn tensor descriptor image_desc with error: " << cudnnGetErrorString(status0);
        return ppl::common::RC_OTHER_ERROR;
    }
    status0 = cudnnDestroyTensorDescriptor(patch_desc0);
    if (status0 != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to destroy the cudnn tensor descriptor patch_desc0 with error: " << cudnnGetErrorString(status0);
        return ppl::common::RC_OTHER_ERROR;
    }
    status0 = cudnnDestroyTensorDescriptor(patch_desc1);
    if (status0 != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to destroy the cudnn tensor descriptor patch_desc1 with error: " << cudnnGetErrorString(status0);
        return ppl::common::RC_OTHER_ERROR;
    }
    status0 = cudnnDestroyFilterDescriptor(filter_desc);
    if (status0 != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to destroy the cudnn filter descriptor filter_desc with error: " << cudnnGetErrorString(status0);
        return ppl::common::RC_OTHER_ERROR;
    }
    status0 = cudnnDestroyConvolutionDescriptor(conv_desc);
    if (status0 != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to destroy the cudnn convolution descriptor conv_desc with error: " << cudnnGetErrorString(status0);
        return ppl::common::RC_OTHER_ERROR;
    }

    return ppl::common::RC_SUCCESS;
}

}}}}} // namespace ppl::nn::llm::cuda::pmx
