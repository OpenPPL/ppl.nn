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

#include "column_parallel_linear_kernel.h"

#include "ppl/common/cuda/nccl_utils.h"
#include "ppl/common/destructor.h"

#include "ppl/kernel/llm/cuda/pmx/i4f16/column_parallel_linear.h"
#include "ppl/kernel/llm/cuda/pmx/column_parallel_linear.h"
#include "ppl/kernel/llm/cuda/pmx/i4f16/quantize.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace opmx {


ppl::common::RetCode I4F16ColumnParallelLinearKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(input, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(weight, 1);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(weight_scale, 2);
    PPLNN_LLM_CUDA_OPTIONAL_INPUT(bias, 3);
    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(output, 0);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [input]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [weight]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(weight);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [weight_scale]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(weight_scale);
    if (bias) {
        PPLNN_LLM_CUDA_DEBUG_TRACE("Input [bias]:\n");
        PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(bias);
    }

    PPLNN_LLM_CUDA_DEBUG_TRACE("in_features: %d\n", param_->in_features);
    PPLNN_LLM_CUDA_DEBUG_TRACE("out_features: %d\n", param_->out_features);
    PPLNN_LLM_CUDA_DEBUG_TRACE("bias_term: %d\n", param_->bias_term);
    PPLNN_LLM_CUDA_DEBUG_TRACE("gather_output: %d\n", param_->gather_output);

    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();

    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(output);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [output]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(output);

    auto input_shape = input->GetShape();
    auto weight_shape = weight->GetShape();
    auto output_shape = output->GetShape();

    if (weight_shape->GetDataType() != ppl::common::DATATYPE_INT4X4) {
        LOG(ERROR) << "currently only support int4x4 weight";
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
    }

    if (ppl::common::DATATYPE_FLOAT16 != input_shape->GetDataType()) {
        LOG(ERROR) << "currently only support fp16";
        return ppl::common::RC_UNSUPPORTED;
    }

    auto nccl_param = GetTensorParallelNcclParam();

    const int64_t M = input_shape->CalcElementsToDimensionExcludingPadding(input_shape->GetDimCount() - 1);
    bool use_fp16_gemm = false;
    {
        const int32_t sm_count = GetCudaDevice()->GetDeviceProp().multiProcessorCount;
        if (sm_count >= 96) {
            use_fp16_gemm = M >= 768;
        } else if (sm_count >= 48) {
            use_fp16_gemm = M >= 512;
        } else {
            use_fp16_gemm = M >= 256;
        }
    }

    uint64_t dequant_weight_buffer_size = 0;
    void *dequant_weight_buffer = nullptr;
    if (use_fp16_gemm) {
        dequant_weight_buffer_size = weight_shape->CalcElementsExcludingPadding() * sizeof(int16_t) * 4;
    }

    uint64_t gather_buffer_size = 0;
    void *gather_buffer = nullptr;
    if (param_->gather_output && nccl_param->size > 1) {
        gather_buffer_size = output_shape->CalcBytesExcludingPadding();
    }

    const int64_t tmp_buffer_size = dequant_weight_buffer_size + gather_buffer_size;
    BufferDesc tmp_buffer_desc;
    auto status = GetCudaDevice()->AllocTmpBuffer(tmp_buffer_size, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_size << "] for kernel[" << GetName()
                << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    ppl::common::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
        GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc);
    });
    gather_buffer = tmp_buffer_desc.addr;

    if (use_fp16_gemm) {
        dequant_weight_buffer = (int8_t*)gather_buffer + gather_buffer_size;

        auto dequant_weight_shape = *weight_shape;
        dequant_weight_shape.SetDim(0, weight_shape->GetDim(0) * 4);
        dequant_weight_shape.SetDataType(weight_scale->GetShape()->GetDataType());

        auto rc = ppl::kernel::llm::cuda::pmx::i4f16::minmax_dequantize_fp16(
            GetStream(),
            weight->GetBufferPtr(),
            weight_scale->GetBufferPtr(),
            dequant_weight_shape.GetDim(0),
            dequant_weight_shape.GetDim(1),
            128,
            dequant_weight_buffer
        );
        if (ppl::common::RC_SUCCESS != rc) {
            return rc;
        }
        const bool use_workspace = GetCudaDevice()->GetSMVersion() >= 90 && M >= 64;
        return ppl::kernel::llm::cuda::pmx::column_parallel_linear(
            GetStream(),
            GetCublasHandle(),
            nullptr,
            input_shape,
            input->GetBufferPtr(),
            &dequant_weight_shape,
            dequant_weight_buffer,
            bias_shape,
            bias_data,
            param_->in_features,
            param_->out_features,
            nccl_param,
            param_->gather_output,
            gather_buffer,
            use_workspace ? GetCudaDevice()->GetCublasWorkspaceSize() : 0,
            use_workspace ? GetCudaDevice()->GetCublasWorkspace() : nullptr,
            output_shape,
            output->GetBufferPtr()
        );
    } else {
        return ppl::kernel::llm::cuda::pmx::i4f16::column_parallel_linear(
            GetStream(),
            GetCudaDevice()->GetI4F16GemmHandle(),
            input_shape,
            input->GetBufferPtr(),
            weight_shape,
            weight->GetBufferPtr(),
            weight_scale->GetBufferPtr(),
            bias_shape,
            bias_data,
            param_->in_features,
            param_->out_features,
            nccl_param,
            param_->gather_output,
            gather_buffer,
            GetCudaDevice()->GetCublasWorkspaceSize(),
            GetCudaDevice()->GetCublasWorkspace(),
            output_shape,
            output->GetBufferPtr()
        );
    }
}

}}}}} // namespace ppl::nn::llm::cuda::opmx
