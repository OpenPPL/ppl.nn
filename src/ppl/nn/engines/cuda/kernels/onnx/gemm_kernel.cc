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

#include "ppl/nn/engines/cuda/kernels/onnx/gemm_kernel.h"

#include "cudakernel/nn/conv/conv_fp16.h"
#include "cudakernel/gemm/gemm.h"

namespace ppl { namespace nn { namespace cuda {

bool GemmKernel::CanDoExecute(const KernelExecContext& ctx) const {
    auto& input = ctx.GetInput<TensorImpl>(0)->GetShape();
    auto& weight = ctx.GetInput<TensorImpl>(1)->GetShape();
    if (input.GetBytesIncludingPadding() == 0) {
        return false;
    }
    if (input.GetDim(1) != weight.GetDim(1)) {
        return false;
    }

    return true;
}

uint64_t GemmKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto A = &ctx.GetInput<TensorImpl>(0)->GetShape();
    return PPLGemmCUDAGetBufSize(A, param_->param.transA);
}

ppl::common::RetCode GemmKernel::DoExecute(KernelExecContext* ctx) {
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
    auto weight = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    // convert filter only if the filter tensor is an output of another kernel
    BufferDesc weight_buffer;
    auto newshape = weight->GetShape();
    if (!param_->extra_param.is_initializer_weight) {    
        auto align_size = 8;
        newshape.SetDim(0, (newshape.GetDim(0) + align_size - 1) / align_size * align_size);

        auto status = GetCudaDevice()->Realloc(newshape, &weight_buffer);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
            return status;
        }
        auto stream = GetStream();
        PPLCUDAGemmModifyWeights(stream, &newshape, weight->GetBufferPtr(), weight_buffer.addr,
                                      &param_->param);
    }
    BufferDescGuard __tmp_buffer_guard__(&weight_buffer, [this](BufferDesc* buffer) {
        GetDevice()->Free(buffer);
    });

    TensorShape bias_shape;
    void* bias = nullptr;
    if (ctx->GetInputCount() >= 3) {
        bias_shape = ctx->GetInput<TensorImpl>(2)->GetShape();
        bias = ctx->GetInput<TensorImpl>(2)->GetBufferPtr();
    }

    fuse_param_t temp_fuse_param;
    ConvertToForwardFuseParam(ctx, GetCudaDevice(), param_->extra_param.fuse_info, temp_fuse_param);

    auto stream = GetStream();
    CUDAModule* module = static_cast<CUDAModule*>(this->GetCommonParam()->module);

    status = PPLCUDAGemmForwardImp(stream, module, &input->GetShape(), input->GetBufferPtr(), &weight->GetShape(),
                                   weight->GetBufferPtr(), bias, &output->GetShape(), output->GetBufferPtr(),
                                   param_->param, tmp_buffer, temp_fuse_param, param_->extra_param.algo_info);

    return status;
}

}}} // namespace ppl::nn::cuda
