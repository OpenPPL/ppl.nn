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

#include "layer_norm_kernel.h"

#include "ppl/kernel/llm/cuda/pmx/layer_norm.h"

#include <iostream>

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode LayerNormKernel::DoExecute(KernelExecContext* ctx) {

    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    // get input, output tensor
    PPLNN_LLM_CUDA_REQUIRED_INPUT(input, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(weight, 1);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(bias, 2);
    PPLNN_LLM_CUDA_OPTIONAL_INPUT(skip_in, 3);

    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(output, 0);
    PPLNN_LLM_CUDA_OPTIONAL_OUTPUT(skip_out, 1);

    // trace info
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [input]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [weight]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(weight);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [bias]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(bias);

    void *skip_in_data = nullptr;
    if (skip_in) {
        PPLNN_LLM_CUDA_DEBUG_TRACE("Input [skip_in]:\n");
        PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(skip_in);
        skip_in_data = skip_in->GetBufferPtr();
    }

    // trace attribute
    PPLNN_LLM_CUDA_DEBUG_TRACE("elementwise_affine: %d\n", param_->elementwise_affine);
    PPLNN_LLM_CUDA_DEBUG_TRACE("eps: %f\n", param_->eps);
    PPLNN_LLM_CUDA_DEBUG_TRACE("axis: %d\n", param_->axis);
    PPLNN_LLM_CUDA_DEBUG_TRACE("skip_term: %d\n", param_->skip_term);

    auto input_shape = input->GetShape();

    if (param_->axis != -1 && param_->axis != input_shape->GetDim(input_shape->GetDimCount() - 1)) {
        LOG(ERROR) << "currently only support axis == -1 or input's last dim.";
        return ppl::common::RC_UNSUPPORTED;
    }

    bool can_trans_input = ctx->IsLastConsumerOfInput(0) && input->GetType() == TENSORTYPE_NORMAL;
    bool can_trans_skip_in = skip_in && ctx->IsLastConsumerOfInput(1) && skip_in->GetType() == TENSORTYPE_NORMAL;

    auto input_data = input->GetBufferPtr();
    if (can_trans_input) {
        output->TransferBufferFrom(input);
    } else {
        PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(output);
    }

    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [output]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(output);

    if (param_->skip_term) {
        if (can_trans_skip_in) {
            skip_out->TransferBufferFrom(skip_in);
        } else {
            PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(skip_out);
        }
        PPLNN_LLM_CUDA_DEBUG_TRACE("Output [skip_out]:\n");
        PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(skip_out);
    }

    if (param_->skip_term && !skip_out) {
        LOG(ERROR) << "skip_out NOT FOUND when skip_term == true.";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (ppl::common::DATATYPE_FLOAT16 != input->GetShape()->GetDataType()) {
        LOG(ERROR) << "currently only support fp16";
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t normalize_shape = input_shape->GetDim(input_shape->GetDimCount() - 1);
    
    return ppl::kernel::llm::cuda::pmx::layer_norm(
        GetStream(),
        input_shape,
        input_data,
        weight->GetBufferPtr(),
        bias->GetBufferPtr(),
        skip_in_data,
        output->GetBufferPtr(),
        skip_out->GetBufferPtr(),
        normalize_shape,
        param_->elementwise_affine,
        param_->eps,
        param_->skip_term);




}

}}}}} // namespace ppl::nn::llm::cuda::pmx