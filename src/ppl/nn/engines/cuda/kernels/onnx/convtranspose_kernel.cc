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

#include "ppl/nn/engines/cuda/kernels/onnx/convtranspose_kernel.h"
#include "ppl/nn/engines/cuda/module/cuda_module.h"
#include "ppl/common/destructor.h"
#include "cudakernel/nn/convtranspose.h"

namespace ppl { namespace nn { namespace cuda {

uint64_t ConvTransposeKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto x = ctx.GetInput<TensorImpl>(0);
    auto y = ctx.GetOutput<TensorImpl>(0);

    return PPLConvTransposeGetBufSizeCuda(x->GetShape(), y->GetShape(), &param_->param);
}

ppl::common::RetCode ConvTransposeKernel::DoExecute(KernelExecContext* ctx) {
    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_bytes = CalcTmpBufferSize(*ctx);
    auto status = GetCudaDevice()->AllocTmpBuffer(tmp_buffer_bytes, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_bytes << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    ppl::common::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
        GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;

    TensorImpl* X = ctx->GetInput<TensorImpl>(0);
    TensorImpl* W = ctx->GetInput<TensorImpl>(1);
    TensorImpl* B = nullptr;
    TensorImpl* Y = ctx->GetOutput<TensorImpl>(0);
    const float* b_data = nullptr;

    if (ctx->GetInputCount() >= 3) {
        B = ctx->GetInput<TensorImpl>(2);
        b_data = B->GetBufferPtr<float>();
    }
    fuse_param_t temp_fuse_param;
    ConvertToForwardFuseParam(ctx, GetCudaDevice(), param_->extra_param.fuse_info, temp_fuse_param);
    CUDAModule* module = static_cast<CUDAModule*>(this->GetCommonParam()->module);

    status = PPLCUDAConvTransposeForward(GetCudaDevice()->GetDeviceId(), GetStream(), module, X->GetShape(),
                                         X->GetBufferPtr(), W->GetBufferPtr(), b_data, Y->GetShape(),
                                         Y->GetBufferPtr(), &param_->param, param_->extra_param.algo_info,
                                         temp_fuse_param, tmp_buffer);

    return status;
}

}}} // namespace ppl::nn::cuda
