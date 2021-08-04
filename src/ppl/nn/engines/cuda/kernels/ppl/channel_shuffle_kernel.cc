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

    if (X->GetShape().GetDimCount() != 4 || Y->GetShape().GetDimCount() != 4) {
        LOG(ERROR) << "incorrect input dimcount: " << X->GetShape().GetDimCount();
        return ppl::common::RC_UNSUPPORTED;
    }
    if (X->GetShape().GetDim(1) % group_) {
        LOG(ERROR) << "unsupported ChanneShuffle group: " << group_;
        return ppl::common::RC_UNSUPPORTED;
    }

    auto Y_shape = Y->GetShape();
    if(Y_shape.GetElementsExcludingPadding() < Y_shape.GetElementsIncludingPadding())
        cudaMemset(Y->GetBufferPtr(), 0, Y_shape.GetBytesIncludingPadding());

    PPLCUDAChannelShuffleForwardImp(GetStream(), group_, &X->GetShape(), X->GetBufferPtr(), &Y->GetShape(),
                                    Y->GetBufferPtr());

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
