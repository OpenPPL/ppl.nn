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

#include "sub_kernel.h"

#include "cudakernel/arithmetic/arithmetic.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace onnx {

ppl::common::RetCode SubKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(input0, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(input1, 1);
    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(output, 0);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [input0]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(input0);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [input1]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(input1);

    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();

    bool can_trans_input0 = ctx->IsLastConsumerOfInput(0)
        && input0->GetType() == TENSORTYPE_NORMAL
        && input0->GetShape()->CalcElementsIncludingPadding() == output->GetShape()->CalcElementsIncludingPadding();

    bool can_trans_input1 = ctx->IsLastConsumerOfInput(1)
        && input1->GetType() == TENSORTYPE_NORMAL
        && input1->GetShape()->CalcElementsIncludingPadding() == output->GetShape()->CalcElementsIncludingPadding();

    auto input0_data = input0->GetBufferPtr();
    auto input1_data = input1->GetBufferPtr();
    if (can_trans_input0) {
        output->TransferBufferFrom(input0);
    } else if (can_trans_input1) {
        output->TransferBufferFrom(input1);
    } else {
        PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(output);
    }
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [output]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(output);

    return PPLCUDAArithMeticSubForwardImp(
        GetStream(),
        input0->GetShape(), input0_data,
        input1->GetShape(), input1_data,
        output->GetShape(), output->GetBufferPtr());
}

}}}}} // namespace ppl::nn::llm::cuda::pmx
