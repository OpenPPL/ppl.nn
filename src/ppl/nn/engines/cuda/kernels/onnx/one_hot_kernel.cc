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

#include "ppl/nn/engines/cuda/kernels/onnx/one_hot_kernel.h"

#include "cudakernel/nn/one_hot.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode OneHotKernel::DoExecute(KernelExecContext* ctx) {
    auto indices = ctx->GetInput<TensorImpl>(0);
    auto depth = ctx->GetInput<TensorImpl>(1);
    auto values = ctx->GetInput<TensorImpl>(2);

    auto output = ctx->GetOutput<TensorImpl>(0);

    // param_->axis;
    uint32_t real_axis = // [-r-1, r]
        param_->axis >= 0 ? param_->axis : param_->axis + indices->GetShape()->GetDimCount() + 1;

    // copy depth and values back to host
    if (depth->GetShape()->GetDimCount() != 1){
        LOG(ERROR)<<"depth of one_hot should be scalar.";
        return ppl::common::RC_INVALID_VALUE;
    }
    if (values->GetShape()->CalcElementsExcludingPadding() != 2) {
        LOG(ERROR) << "value tensor should be [off_value, on_value] ";
        return ppl::common::RC_INVALID_VALUE;
    }

    ppl::common::RetCode status = PPLCUDAOneHotForwardImp(GetStream(), indices->GetBufferPtr(), 
                                                        values->GetShape(), values->GetBufferPtr(), 
                                                        output->GetShape(), output->GetBufferPtr(), real_axis);

    return status;
}

}}} // namespace ppl::nn::cuda