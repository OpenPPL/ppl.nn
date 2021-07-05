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

#include "ppl/nn/engines/cuda/kernels/onnx/scatter_nd_kernel.h"

#include "cudakernel/memory/scatter_nd.h"

namespace ppl { namespace nn { namespace cuda {

bool ScatterNdKernel::CanDoExecute(const KernelExecContext& ctx) const {
    return ctx.GetInput<TensorImpl>(0)->GetShape().GetBytesIncludingPadding() != 0;
}

uint64_t ScatterNdKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto input = ctx.GetInput<TensorImpl>(0);
    auto indices = ctx.GetInput<TensorImpl>(1);
    return PPLScatterNDGetTempBufferSize(&input->GetShape(), input->GetBufferPtr(), &indices->GetShape(),
                                         indices->GetBufferPtr());
}

ppl::common::RetCode ScatterNdKernel::DoExecute(KernelExecContext* ctx) {
    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_bytes = CalcTmpBufferSize(*ctx);
    auto status = GetCudaDevice()->AllocTmpBuffer(tmp_buffer_bytes, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_bytes << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    BufferDescGuard __tmp_buffer_guard(&tmp_buffer_desc, [this](BufferDesc* buffer) -> void {
        GetCudaDevice()->FreeTmpBuffer(buffer);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;

    auto input = ctx->GetInput<TensorImpl>(0);
    auto indices = ctx->GetInput<TensorImpl>(1);
    auto updates = ctx->GetInput<TensorImpl>(2);
    auto output = ctx->GetOutput<TensorImpl>(0);

    status = PPLCUDAScatterNDForwardImp(GetStream(), &input->GetShape(), input->GetBufferPtr(), &indices->GetShape(),
                                        indices->GetBufferPtr(), &updates->GetShape(), updates->GetBufferPtr(),
                                        &output->GetShape(), output->GetBufferPtr(), tmp_buffer);
    return status;
}

}}} // namespace ppl::nn::cuda
