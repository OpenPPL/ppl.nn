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

#include "online_dequantize_swiglu_quantize_kernel.h"

#include "ppl/kernel/llm/cuda/pmx/i8i8/quantize.h"

#include "ppl/common/destructor.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace opmx {

ppl::common::RetCode I8I8OnlineDequantizeSwiGLUQuantizeKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(input, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(input_scale_outer, 1);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(input_scale_inner, 2);

    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(output, 0);
    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(scale, 1);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [input]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [input_scale_outer]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(input_scale_outer);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [input_scale_inner]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(input_scale_inner);

    PPLNN_LLM_CUDA_DEBUG_TRACE("beta: %f\n", param_->beta);

    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();

    if (input_scale_outer->GetShape()->GetDataType() != input_scale_inner->GetShape()->GetDataType()) {
        LOG(ERROR) << "datatype of scale_outer must be equal to datatype of scale_inner: "
            << ppl::common::GetDataTypeStr(input_scale_outer->GetShape()->GetDataType()) << " vs. "
            << ppl::common::GetDataTypeStr(input_scale_inner->GetShape()->GetDataType());
        return ppl::common::RC_INVALID_VALUE;
    }

    if (input_scale_outer->GetShape()->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "currently only support dequantize to fp16";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (input->GetShape()->GetDataType() != ppl::common::DATATYPE_INT32) {
        LOG(ERROR) << "currently only support dequantize int32 data";
        return ppl::common::RC_UNSUPPORTED;
    }

    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(output);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [output]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(output);

    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(scale);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [scale]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(scale);

    const int64_t batch = input_scale_outer->GetShape()->CalcElementsIncludingPadding();
    const int64_t dequant_dim = input_scale_inner->GetShape()->CalcElementsIncludingPadding();
    const int64_t total_elem = input->GetShape()->CalcElementsIncludingPadding();
    if (total_elem != batch * dequant_dim) {
        LOG(ERROR) << "input.numel must be equal to scale_outer.numel * scale_inner.numel): "
            << batch << " * " << dequant_dim << " != " << total_elem;
        return ppl::common::RC_INVALID_VALUE;
    }

    uint64_t dequant_buffer_size = output->GetShape()->CalcElementsIncludingPadding() * sizeof(int16_t);
    void *dequant_buffer = nullptr;

    BufferDesc tmp_buffer_desc;
    auto status = GetCudaDevice()->AllocTmpBuffer(dequant_buffer_size, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << dequant_buffer_size << "] for kernel[" << GetName()
                << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    ppl::common::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
        GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc);
    });
    dequant_buffer = tmp_buffer_desc.addr;

    auto tensor_layout = GetEngineOptions().cublas_layout_hint == CUBLAS_LAYOUT_AMPERE
        ? ppl::kernel::llm::cuda::MATRIX_LAYOUT_COL32
        : ppl::kernel::llm::cuda::MATRIX_LAYOUT_ROW_MAJOR;

    const int64_t dim_count = output->GetShape()->GetDimCount();
    const int64_t quant_dim = output->GetShape()->GetDim(dim_count - 1);
    return ppl::kernel::llm::cuda::pmx::i8i8::minmax_requantize_swiglu_fp16(
        GetStream(),
        input->GetBufferPtr(),
        input_scale_outer->GetBufferPtr(),
        input_scale_inner->GetBufferPtr(),
        batch,
        quant_dim,
        param_->beta,
        tensor_layout,
        ppl::kernel::llm::cuda::pmx::i8i8::token_up_scale,
        ppl::kernel::llm::cuda::pmx::i8i8::token_down_scale,
        ppl::kernel::llm::cuda::pmx::i8i8::hidden_down_scale,
        dequant_buffer,
        output->GetBufferPtr(),
        scale->GetBufferPtr()
    );
}

}}}}} // namespace ppl::nn::llm::cuda::opmx
