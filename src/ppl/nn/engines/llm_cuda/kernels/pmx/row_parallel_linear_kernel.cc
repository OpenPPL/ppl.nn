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

#include "row_parallel_linear_kernel.h"

#include "ppl/common/cuda/nccl_utils.h"
#include "ppl/common/destructor.h"

#include "cudakernel/llm/row_parallel_linear.h"
#include "cudakernel/gemm/gemm.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode RowParallelLinearKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(input, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(weight, 1);
    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(output, 0);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [input]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [weight]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(weight);

    PPLNN_LLM_CUDA_DEBUG_TRACE("in_features: %d\n", param_->in_features);
    PPLNN_LLM_CUDA_DEBUG_TRACE("out_features: %d\n", param_->out_features);
    PPLNN_LLM_CUDA_DEBUG_TRACE("bias_term: %d\n", param_->bias_term);
    PPLNN_LLM_CUDA_DEBUG_TRACE("input_is_parallel: %d\n", param_->input_is_parallel);

    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(output);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [output]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(output);

    auto input_shape = input->GetShape();
    auto weight_shape = weight->GetShape();
    auto output_shape = output->GetShape();

    if (ppl::common::DATATYPE_FLOAT16 != input_shape->GetDataType()) {
        LOG(ERROR) << "currently only support fp16";
        return ppl::common::RC_UNSUPPORTED;
    }

    GemmKernelParam gemm_param{1, 0, false, true};
    auto cublas_handle = GetCublasHandle();
    auto nccl_param = GetTensorParallelNcclParam();
    cublasLtMatmulAlgo_t cublas_algo;

    if (param_->bias_term) {
        LOG(ERROR) << "bias_term unsupported";
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t M = input_shape->CalcElementsToDimensionExcludingPadding(input_shape->GetDimCount() - 1);
    const bool use_workspace = M >= 64;

    return PPLCUDARowParallelLinearForwardImp(
        GetStream(), cublas_handle, gemm_param, nccl_param,
        input_shape, input->GetBufferPtr(), nullptr,
        weight_shape, weight->GetBufferPtr(),
        output_shape, output->GetBufferPtr(),
        nullptr, nullptr, 1,
        use_workspace ? GetCudaDevice()->GetCubalsWorkspace() : nullptr,
        use_workspace ? GetCudaDevice()->GetCublasWorkspaceSize() : 0,
        false, param_->input_is_parallel, cublas_algo);

}

}}}}} // namespace ppl::nn::llm::cuda::pmx
