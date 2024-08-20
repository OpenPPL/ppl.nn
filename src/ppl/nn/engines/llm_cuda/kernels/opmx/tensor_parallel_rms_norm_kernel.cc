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

#include "tensor_parallel_rms_norm_kernel.h"

#include "ppl/kernel/llm/cuda/pmx/tensor_parallel_rms_norm.h"
#include "ppl/common/destructor.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace opmx {


ppl::common::RetCode TensorParallelRMSNormKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(input, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(weight, 1);

    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(output, 0);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [input]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [weight]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(weight);

    PPLNN_LLM_CUDA_DEBUG_TRACE("eps: %f\n", param_->eps);
    PPLNN_LLM_CUDA_DEBUG_TRACE("axis: %d\n", param_->axis);
    PPLNN_LLM_CUDA_DEBUG_TRACE("scale: %d\n", param_->scale);

    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();

    auto input_shape = input->GetShape();

    if (param_->axis != -1 && param_->axis != input_shape->GetDim(input_shape->GetDimCount() - 1)) {
        LOG(ERROR) << "currently only support axis == -1 or input's last dim.";
        return ppl::common::RC_UNSUPPORTED;
    }

    bool can_trans_input = ctx->IsLastConsumerOfInput(0) && input->GetType() == TENSORTYPE_NORMAL;

    auto input_data = input->GetBufferPtr();
    if (can_trans_input) {
        output->TransferBufferFrom(input);
    } else {
        PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(output);
    }
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [output]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(output);

    if (ppl::common::DATATYPE_FLOAT16 != input->GetShape()->GetDataType()) {
        LOG(ERROR) << "currently only support fp16";
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t dim_count = input_shape->GetDimCount();
    const int64_t real_axis = param_->axis > 0 ? param_->axis : (param_->axis + dim_count);

    const int64_t batch = input_shape->CalcElementsToDimensionIncludingPadding(real_axis);
    const int64_t norm_dim = input_shape->CalcElementsFromDimensionIncludingPadding(real_axis);

    int workspace_size = batch * sizeof(float);
    BufferDesc tmpbuffer_desc;
    auto status = GetCudaDevice()->AllocTmpBuffer(workspace_size, &tmpbuffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << workspace_size << "] for kernel[" << GetName()
                << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    ppl::common::Destructor tp_pow_sum_tmpbuffer_guard([this, &tmpbuffer_desc]() -> void {
        GetCudaDevice()->FreeTmpBuffer(&tmpbuffer_desc);
    });
    void* tp_pow_sum = tmpbuffer_desc.addr;

    auto nccl_param = GetTensorParallelNcclParam();

    return ppl::kernel::llm::cuda::pmx::tensor_parallel_rms_norm_fp16(
        GetStream(),
        input_data,
        weight->GetBufferPtr(),
        param_->eps,
        param_->scale,
        batch,
        norm_dim,
        nccl_param,
        tp_pow_sum,
        output->GetBufferPtr()
    );
}

}}}}} // namespace ppl::nn::llm::cuda::opmx
