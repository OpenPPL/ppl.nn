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

#include "split_kernel.h"

#include "cudakernel/memory/split.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace onnx {


ppl::common::RetCode SplitKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(input, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(split, 1);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [input]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [split]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(split);

    PPLNN_LLM_CUDA_DEBUG_TRACE("axis: %d\n", param_->axis);

    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();

    const int32_t axis = param_->axis < 0 ? param_->axis + input->GetShape()->GetDimCount() : param_->axis;
    bool can_trans = ctx->IsLastConsumerOfInput(0)
        && input->GetType() == TENSORTYPE_NORMAL
        && ctx->GetOutputCount() == 1
        && input->GetShape()->CalcElementsIncludingPadding() == ctx->GetOutput<TensorImpl>(0)->GetShape()->CalcElementsIncludingPadding();

    if (can_trans) {
        auto output = ctx->GetOutput<TensorImpl>(0);
        output->TransferBufferFrom(input);
        PPLNN_LLM_CUDA_DEBUG_TRACE("Output [output]:\n");
        PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(output);
        return ppl::common::RC_SUCCESS;
    }

    std::vector<const TensorShape*> dst_shapes(ctx->GetOutputCount());
    std::vector<void*> dst_datas(ctx->GetOutputCount());

    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto output = ctx->GetOutput<TensorImpl>(i);
        PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(output);
        PPLNN_LLM_CUDA_DEBUG_TRACE("Output [outputs[%u]]:\n", i);
        PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(output);
        dst_datas[i] = output->GetBufferPtr();
        dst_shapes[i] = output->GetShape();
    }

    ppl::common::RetCode rc = ppl::common::RC_UNSUPPORTED;

    if (ctx->GetOutputCount() == 3) {
        rc = PPLCUDAAlignedSplit3ForwardImp(
            GetStream(),
            input->GetShape(),
            input->GetBufferPtr(),
            axis,
            dst_shapes[0], dst_datas[0],
            dst_shapes[1], dst_datas[1],
            dst_shapes[2], dst_datas[2]);
    }

    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "currently only support split 3 with inner dims aligned with 8.";
    }
    return rc;
}

}}}}} // namespace ppl::nn::llm::cuda::pmx
