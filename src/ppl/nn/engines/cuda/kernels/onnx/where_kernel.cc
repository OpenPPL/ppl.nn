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

#include "ppl/nn/engines/cuda/kernels/onnx/where_kernel.h"

#include "cudakernel/memory/where.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode WhereKernel::DoExecute(KernelExecContext* ctx) {
    auto cond = ctx->GetInput<TensorImpl>(0);
    auto x = ctx->GetInput<TensorImpl>(1);
    auto y = ctx->GetInput<TensorImpl>(2);
    auto output = ctx->GetOutput<TensorImpl>(0);

    ppl::common::RetCode status = PPLCUDAWhereForwardImp(
        GetStream(), &cond->GetShape(), (const bool*)cond->GetBufferPtr(), &x->GetShape(), x->GetBufferPtr(),
        &y->GetShape(), y->GetBufferPtr(), &output->GetShape(), output->GetBufferPtr());

    return status;
}

}}} // namespace ppl::nn::cuda
