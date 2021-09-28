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

#include "ppl/nn/engines/cuda/kernels/onnx/reduce_sum_kernel.h"

#include <numeric>

#include "cudakernel/reduce/reduce.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode ReduceSumKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    ReduceParam param = ReduceSum;
    auto input_shape = input->GetShape();
    uint32_t n_outer = 1, n_reduce = 1, n_inner = 1;

    const uint32_t dim_count = input_shape.GetDimCount();
    if (param_->axes.empty()) { // empty axes means reduce all dims
        n_reduce =
            accumulate(input_shape.GetDims(), input_shape.GetDims() + dim_count, n_reduce, std::multiplies<uint32_t>());
    } else {
        // uint32_t real_axis[param_->axes.size()];
        std::vector<uint32_t> real_axis(param_->axes.size());

        for (uint32_t i = 0; i < param_->axes.size(); ++i) {
            real_axis[i] = (param_->axes[i] + dim_count) % dim_count;
            if (i > 0 && real_axis[i] != real_axis[i - 1] + 1) {
                return ppl::common::RC_UNSUPPORTED;
            }
            n_reduce *= input_shape.GetDim(real_axis[i]);
        }
        n_outer = accumulate(input_shape.GetDims(), input_shape.GetDims() + real_axis[0], n_outer,
                             std::multiplies<uint32_t>());
        n_inner = accumulate(input_shape.GetDims() + real_axis[param_->axes.size() - 1] + 1,
                             input_shape.GetDims() + dim_count, n_inner, std::multiplies<uint32_t>());
    }

    PPLReduceDimDes des(n_inner, n_reduce, n_outer);
    ppl::common::RetCode status =
        PPLCUDAReduceForwardImp(GetStream(), param, des, &input->GetShape(), input->GetBufferPtr(), &output->GetShape(),
                                output->GetBufferPtr());
    return status;
}

}}} // namespace ppl::nn::cuda
