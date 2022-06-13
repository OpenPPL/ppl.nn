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

#include "ppl/nn/engines/cuda/kernels/onnx/clip_kernel.h"

#include <float.h>

#include "cudakernel/unary/clip.h"

namespace ppl { namespace nn { namespace cuda {

bool ClipKernel::CanDoExecute(const KernelExecContext& ctx) const {
    auto tensor = ctx.GetInput<TensorImpl>(0);
    if (!tensor || tensor->GetShape()->GetBytesIncludingPadding() == 0) {
        return false;
    }
    return true;
}

ppl::common::RetCode ClipKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    auto clip_min = ctx->GetInput<TensorImpl>(1);
    auto clip_max = ctx->GetInput<TensorImpl>(2);
    float min_val = std::numeric_limits<float>::lowest();
    float max_val = std::numeric_limits<float>::max();

    if (clip_min) {
        auto status = clip_min->CopyToHost(&min_val);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "get min value of clip failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }
    }
    if (clip_max) {
        auto status = clip_max->CopyToHost(&max_val);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "get max value of clip failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }
    }

    ppl::common::RetCode status = PPLCUDAClipForwardImp(GetStream(), input->GetShape(), input->GetBufferPtr(),
                                                        output->GetShape(), output->GetBufferPtr(), min_val, max_val);
    return status;
}

}}} // namespace ppl::nn::cuda
