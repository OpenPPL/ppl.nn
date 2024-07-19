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

#include "pixel_unshuffle_kernel.h"

#include "ppl/common/destructor.h"

#include "ppl/kernel/llm/cuda/pmx/pixel_unshuffle.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace opmx {

ppl::common::RetCode PixelUnshuffleKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(input, 0);
    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(output, 0);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [input]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(input);

    PPLNN_LLM_CUDA_DEBUG_TRACE("scale_factor: %d\n", param_->scale_factor);
    PPLNN_LLM_CUDA_DEBUG_TRACE("data_layout: %d\n", param_->data_layout);

    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();

    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(output);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [output]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(output);

    auto input_shape = input->GetShape();

    if (ppl::common::DATATYPE_FLOAT16 != input_shape->GetDataType()) {
        LOG(ERROR) << "currently only support fp16";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (param_->data_layout != param_->DATA_LAYOUT_NHWC) {
        LOG(ERROR) << "currently only support scaling_type == 'nhwc'";
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::kernel::llm::cuda::pmx::pixel_unshuffle(
        GetStream(),
        input_shape,
        input->GetBufferPtr(),
        param_->scale_factor,
        output->GetBufferPtr()
    );

}

}}}}} // namespace ppl::nn::llm::cuda::opmx
