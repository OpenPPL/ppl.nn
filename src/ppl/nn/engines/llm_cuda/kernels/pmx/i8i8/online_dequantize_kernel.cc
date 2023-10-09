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

#include "online_dequantize_kernel.h"

#include "ppl/kernel/llm/cuda/pmx/i8i8/quantize.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode I8I8OnlineDequantizeKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(input, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(scale_outer, 1);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(scale_inner, 2);
    PPLNN_LLM_CUDA_OPTIONAL_INPUT(bias, 3);

    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(output, 0);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [input]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [scale_outer]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(scale_outer);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [scale_inner]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(scale_inner);
    if (bias) {
        PPLNN_LLM_CUDA_DEBUG_TRACE("Input [bias]:\n");
        PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(bias);
    }

    PPLNN_LLM_CUDA_DEBUG_TRACE("bias_term: %d\n", param_->bias_term);

    if (scale_outer->GetShape()->GetDataType() != scale_inner->GetShape()->GetDataType()) {
        LOG(ERROR) << "datatype of scale_outer must be equal to datatype of scale_inner: "
            << ppl::common::GetDataTypeStr(scale_outer->GetShape()->GetDataType()) << " vs. "
            << ppl::common::GetDataTypeStr(scale_inner->GetShape()->GetDataType());
        return ppl::common::RC_INVALID_VALUE;
    }

    if (scale_outer->GetShape()->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "currently only support dequantize to fp16";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (input->GetShape()->GetDataType() != ppl::common::DATATYPE_INT32) {
        LOG(ERROR) << "currently only support dequantize int32 data";
        return ppl::common::RC_UNSUPPORTED;
    }

    TensorShape *bias_shape = nullptr;
    void *bias_data = nullptr;
    if (param_->bias_term) {
        if (!bias) {
            LOG(ERROR) << "bias_term == true but bias not found.";
            return ppl::common::RC_NOT_FOUND;
        }
        bias_shape = bias->GetShape();
        bias_data = bias->GetBufferPtr();

        if (bias_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
            LOG(ERROR) << "currently only support fp16 bias";
            return ppl::common::RC_UNSUPPORTED;
        }
    }

    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(output);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [output]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(output);

    const int64_t batch = scale_outer->GetShape()->CalcElementsIncludingPadding();
    const int64_t quant_dim = scale_inner->GetShape()->CalcElementsIncludingPadding();
    const int64_t total_elem = input->GetShape()->CalcElementsIncludingPadding();
    if (total_elem != batch * quant_dim) {
        LOG(ERROR) << "input.numel must be equal to scale_outer.numel * scale_inner.numel): "
            << batch << " * " << quant_dim << " != " << total_elem;
        return ppl::common::RC_INVALID_VALUE;
    }

    return ppl::kernel::llm::cuda::pmx::i8i8::minmax_dequantize_fp16(
        GetStream(),
        input->GetBufferPtr(),
        bias_data,
        scale_outer->GetBufferPtr(),
        scale_inner->GetBufferPtr(),
        batch,
        quant_dim,
        ppl::kernel::llm::cuda::pmx::i8i8::token_down_scale,
        ppl::kernel::llm::cuda::pmx::i8i8::hidden_down_scale,
        ppl::kernel::llm::cuda::MATRIX_LAYOUT_COL32,
        output->GetBufferPtr()
    );
}

}}}}} // namespace ppl::nn::llm::cuda::pmx
