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

#include "ppl/nn/engines/cuda/kernels/onnx/matmul_kernel.h"
#include "ppl/common/destructor.h"
#include "cudakernel/nn/conv/conv_fp16.h"
#include "cudakernel/gemm/bgemm.h"

namespace ppl { namespace nn { namespace cuda {

bool MatMulKernel::CanDoExecute(const KernelExecContext& ctx) const {
    const TensorShape& input0 = *ctx.GetInput<TensorImpl>(0)->GetShape();
    const TensorShape& input1 = *ctx.GetInput<TensorImpl>(1)->GetShape();
    if (input0.CalcBytesIncludingPadding() == 0) {
        return false;
    }
    if (input1.CalcBytesIncludingPadding() == 0) {
        return false;
    }
    // K must be the same
    uint32_t dim_count0 = input0.GetDimCount();
    uint32_t dim_count1 = input1.GetDimCount();
    if (input0.GetDim(dim_count0 - 1) != input1.GetDim(dim_count1 - 2)) {
        return false;
    }

    return true;
}

uint64_t MatMulKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    // TODO
    auto A = ctx.GetInput<TensorImpl>(0)->GetShape();
    return PPLBgemmCUDAGetBufSize(A, param_->param.transA);
}

ppl::common::RetCode MatMulKernel::DoExecute(KernelExecContext* ctx) {
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

    auto input0 = ctx->GetInput<TensorImpl>(0);
    auto weight = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    // convert filter only if the filter tensor is an output of another kernel
    BufferDesc weight_buffer;
    auto newshape = *weight->GetShape();
    {
        auto align_size = 8;
        auto dim_count = newshape.GetDimCount();
        newshape.SetDim(dim_count - 2, (newshape.GetDim(dim_count - 2) + align_size - 1) / align_size * align_size);

        auto status = GetCudaDevice()->Realloc(newshape, &weight_buffer);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
            return status;
        }
        auto stream = GetStream();
        PPLCUDABgemmModifyWeights(stream, weight->GetShape(), weight->GetBufferPtr(), weight_buffer.addr,
                                  &param_->param);
    }
    ppl::common::Destructor __tmp_buffer_guard__([this, &weight_buffer]() -> void {
        GetCudaDevice()->Free(&weight_buffer);
    });

    BufferDesc input0_buffer;
    auto newshape0 = *input0->GetShape();
    auto dim_count = newshape0.GetDimCount();
    auto K = newshape0.GetDim(dim_count - 1);
    auto align_size = 8;
    auto K_pad = (K + align_size - 1) / align_size * align_size;
    bool is_input0_pad = K != K_pad;
    void* bmm_input0;
    if (is_input0_pad) {
        newshape0.SetDim(dim_count - 1, K_pad);
        auto status = GetCudaDevice()->Realloc(newshape0, &input0_buffer);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
            return status;
        }
        auto stream = GetStream();
        PPLCUDABgemmPadInput(stream, input0->GetShape(), input0->GetBufferPtr(), input0_buffer.addr, &param_->param);
        bmm_input0 = input0_buffer.addr;
    } else {
        bmm_input0 = input0->GetBufferPtr();
    }
    ppl::common::Destructor __input0_buffer_guard__([this, &input0_buffer]() -> void {
        GetCudaDevice()->Free(&input0_buffer);
    });

    auto newshape_out = *output->GetShape();
    auto out_dim_count = newshape_out.GetDimCount();
    auto N = newshape_out.GetDim(out_dim_count - 1);
    auto N_pad = (N + align_size - 1) / align_size * align_size;
    BufferDesc output_buffer;
    bool is_output_pad = N != N_pad;
    void* bgemm_out;
    if (is_output_pad) {
        newshape_out.SetDim(out_dim_count - 1, N_pad);
        auto status = GetCudaDevice()->Realloc(newshape_out, &output_buffer);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
            return status;
        }
        bgemm_out = output_buffer.addr;
    } else {
        bgemm_out = output->GetBufferPtr();
    }
    ppl::common::Destructor __output_buffer_guard__([this, &output_buffer]() -> void {
        GetCudaDevice()->Free(&output_buffer);
    });

    fuse_param_t temp_fuse_param;
    ConvertToForwardFuseParam(ctx, GetCudaDevice(), param_->extra_param.fuse_info, temp_fuse_param);

    auto stream = GetStream();
    int device_id = GetDeviceId();
    CUDAModule* module = static_cast<CUDAModule*>(this->GetCommonParam()->module);

    const TensorShape& shape_in0 = *input0->GetShape();

    if (shape_in0.GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        status = PPLCUDABgemmForwardImp(device_id, stream, module, input0->GetShape(), bmm_input0,
                                        weight->GetShape(), weight_buffer.addr, output->GetShape(), bgemm_out,
                                        param_->param, tmp_buffer, temp_fuse_param, param_->extra_param.algo_info);
    }

    if (is_output_pad) {
        PPLCUDABgemmCvtOutput(stream, output->GetShape(), output->GetBufferPtr(), bgemm_out);
    }

    return status;
}

}}} // namespace ppl::nn::cuda
