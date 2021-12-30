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

#include "ppl/nn/engines/cuda/kernels/onnx/avepooling_kernel.h"

#include "cudakernel/nn/pooling_ave.h"
#include "cudakernel/nn/global_pooling_ave.h"
#include<iostream>

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode AvePoolingKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    ppl::common::RetCode status = ppl::common::RC_UNSUPPORTED;
    auto input_id = input->GetEdge()->GetId();
    auto input_quant = GetCommonParam()->cuda_tensor_info->at(input_id);
    auto output_id = output->GetEdge()->GetId();
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output_id);
    if (param_->global_pooling) {
        status = PPLCUDAGlobalAvePoolingForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(),
                                                   &output->GetShape(), output->GetBufferPtr(), input_quant.scale[0], output_quant.scale[0]);
    } else {
        int32_t kernel_h = param_->kernel_shape[0];
        int32_t kernel_w = param_->kernel_shape[1];
        int32_t stride_h = param_->strides[0];
        int32_t stride_w = param_->strides[1];
        int32_t pad_h = param_->pads.size() >= 1 ? param_->pads[0] : 0;
        int32_t pad_w = param_->pads.size() >= 2 ? param_->pads[1] : 0;
        // 1*1 pooling, just transfer
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && pad_h == 0 && pad_w == 0) {
            output->TransferBufferFrom(input);
            return ppl::common::RC_SUCCESS;
        }

        int32_t if_excluding_padding = 1;
        if (param_->mode == ppl::nn::common::PoolingParam::POOLING_AVERAGE_INCLUDE) {
            if_excluding_padding = 0;
        }
        status = PPLCUDAAvePoolingForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(),
                                             &output->GetShape(), output->GetBufferPtr(), kernel_h, kernel_w, stride_h,
                                             stride_w, pad_h, pad_w, if_excluding_padding, input_quant.scale[0], output_quant.scale[0]);
    }
    return status;
}

}}} // namespace ppl::nn::cuda
