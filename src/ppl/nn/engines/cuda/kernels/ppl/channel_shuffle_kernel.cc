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

#include "ppl/nn/engines/cuda/kernels/ppl/channel_shuffle_kernel.h"

#include "cudakernel/memory/channel_shuffle.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode ChannelShuffleKernel::DoExecute(KernelExecContext* ctx) {
    auto X = ctx->GetInput<TensorImpl>(0);
    auto Y = ctx->GetOutput<TensorImpl>(0);
    int group_ = param_->group;

    auto input_id0 = X->GetEdge()->GetId();
    auto input_quant0 = GetCommonParam()->cuda_tensor_info->at(input_id0);
    auto output_id0 = Y->GetEdge()->GetId();
    auto output_quant0 = GetCommonParam()->cuda_tensor_info->at(output_id0);

    if (X->GetShape().GetDimCount() != 4 || Y->GetShape().GetDimCount() != 4) {
        LOG(ERROR) << "incorrect input dimcount: " << X->GetShape().GetDimCount();
        return ppl::common::RC_UNSUPPORTED;
    }
    if (X->GetShape().GetDim(1) % group_) {
        LOG(ERROR) << "unsupported ChanneShuffle group: " << group_;
        return ppl::common::RC_UNSUPPORTED;
    }

    if (ctx->GetOutputCount() == 1) {
        auto Y_shape = Y->GetShape();
        PPLCUDAChannelShuffleForwardImp(GetStream(), group_, &X->GetShape(), X->GetBufferPtr(),
                                                             &Y->GetShape(), Y->GetBufferPtr(), 
                                                             input_quant0.scale[0], output_quant0.scale[0]);
    }

    if (ctx->GetOutputCount() == 2) {
        auto X2 = ctx->GetInput<TensorImpl>(1);
        auto Y2 = ctx->GetOutput<TensorImpl>(1);
        auto input_id1 = X2->GetEdge()->GetId();
        auto input_quant1 = GetCommonParam()->cuda_tensor_info->at(input_id1);
        auto output_id1 = Y2->GetEdge()->GetId();
        auto output_quant1 = GetCommonParam()->cuda_tensor_info->at(output_id1);
        PPLCUDAFuseChannelShuffleForwardImp(GetStream(), group_, &X->GetShape(), X->GetBufferPtr(), X2->GetBufferPtr(),
                                                                 &Y->GetShape(), Y->GetBufferPtr(), Y2->GetBufferPtr(),
                                                                 input_quant0.scale[0], input_quant1.scale[0],
                                                                 output_quant0.scale[0], output_quant1.scale[0]);
    }
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
