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

#include "ppl/nn/engines/cuda/kernels/onnx/batch_normalization_kernel.h"

#include "cudakernel/nn/batch_normalization.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode BatchNormalizationKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto scale = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);
    auto input_id = input->GetEdge()->GetId();
    auto input_quant = GetCommonParam()->cuda_tensor_info->at(input_id);

    // How to use:
    auto has_relu = param_->extra_param.has_relu;

    auto output_id = output->GetEdge()->GetId();
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output_id);
    ppl::common::RetCode status = PPLCUDABatchNormalizationForwardImp(
        GetStream(), &input->GetShape(), input->GetBufferPtr(), &scale->GetShape(), scale->GetBufferPtr(),
        ctx->GetInput<TensorImpl>(2)->GetBufferPtr(), ctx->GetInput<TensorImpl>(3)->GetBufferPtr(),
        ctx->GetInput<TensorImpl>(4)->GetBufferPtr(), &output->GetShape(), output->GetBufferPtr(), param_->param.epsilon,
        input_quant.scale[0], 1.0f / output_quant.scale[0], has_relu);
    return status;
}

}}} // namespace ppl::nn::cuda
