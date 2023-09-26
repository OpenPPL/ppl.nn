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

#include "online_quantize_kernel.h"

#include "ppl/kernel/llm/cuda/pmx/i8i8/quantize.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode I8I8OnlineQuantizeKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(input, 0);

    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(output, 0);
    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(scale, 1);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [input]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(input);

    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(output);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [output]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(scale);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [scale]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(scale);

    if (input->GetShape()->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "currently only support fp16 input";
        return ppl::common::RC_UNSUPPORTED;
    }

    const auto dim_count = input->GetShape()->GetDimCount();
    const int64_t quant_dim = input->GetShape()->GetDim(dim_count - 1);
    const int64_t batch = input->GetShape()->CalcElementsToDimensionIncludingPadding(dim_count - 1);

    return ppl::kernel::llm::cuda::pmx::i8i8::minmax_quantize_fp16(
        GetStream(),
        input->GetBufferPtr(),
        batch,
        quant_dim,
        ppl::kernel::llm::cuda::pmx::i8i8::token_up_scale,
        ppl::kernel::llm::cuda::MATRIX_LAYOUT_COL32,
        output->GetBufferPtr(),
        scale->GetBufferPtr()
    );
}

}}}}} // namespace ppl::nn::llm::cuda::pmx
