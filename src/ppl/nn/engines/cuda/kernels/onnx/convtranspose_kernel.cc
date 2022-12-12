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

#define ALGO_MAX_TIME (3.0e+10)

#define ALLOC_BUFFERF_FOR_ALGO_SELECT(___buffer_name___, ___size___, ___ret___)                  \
    BufferDesc ___buffer_name___;  \
    status =  GetCudaDevice()->ReallocWithRandomValue(___size___, &___buffer_name___);             \
    if (status != RC_SUCCESS) {                                                                  \
        LOG(DEBUG) << "alloc " #___buffer_name___ " tensor failed";                              \
        return ppl::common::RC_UNSUPPORTED;                                                                        \
    }                                                                                            \
    ppl::common::Destructor __##___buffer_name___##_guard__([this, &___buffer_name___]() -> void { \
        GetCudaDevice()->Free(&___buffer_name___);                                                \
    });

uint64_t ConvTransposeKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto x = ctx.GetInput<TensorImpl>(0);
    auto y = ctx.GetOutput<TensorImpl>(0);

    ConvTransposeKernelParam param_kernel_;
    param_kernel_.auto_pad = param_->param.auto_pad;
    param_kernel_.group = param_->param.group;
    param_kernel_.dilations = param_->param.dilations;
    param_kernel_.kernel_shape = param_->param.kernel_shape;
    param_kernel_.pads = param_->param.pads;
    param_kernel_.strides = param_->param.strides;
    param_kernel_.output_padding = param_->param.output_padding;
    param_kernel_.output_shape = param_->param.output_shape;

    return PPLConvTransposeGetBufSizeCuda(x->GetShape(), y->GetShape(), &param_kernel_);
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
    ConvTransposeKernelParam param_kernel_;
    param_kernel_.auto_pad = param_->param.auto_pad;
    param_kernel_.group = param_->param.group;
    param_kernel_.dilations = param_->param.dilations;
    param_kernel_.kernel_shape = param_->param.kernel_shape;
    param_kernel_.pads = param_->param.pads;
    param_kernel_.strides = param_->param.strides;
    param_kernel_.output_padding = param_->param.output_padding;
    param_kernel_.output_shape = param_->param.output_shape;

    // convert filter only if the filter tensor is an output of another kernel
    BufferDesc weight_buffer;
    auto newshape = *W->GetShape();
    if (!param_->extra_param.is_initializer_weight) {
        auto align_size = 8;
        int stride_h = param_->param.strides[0];
        int stride_w = param_->param.strides[1];
        int kernel_u = (newshape.GetDim(2) + stride_h - 1) / stride_h;
        int kernel_v = (newshape.GetDim(3) + stride_w - 1) / stride_w;
        int pattern_num = stride_h * stride_w;
        newshape.SetDim(0, (newshape.GetDim(0) + align_size - 1) / align_size * align_size);
        newshape.SetPadding1(1, (newshape.GetDim(1) + align_size - 1) / align_size * align_size - newshape.GetDim(1));
        newshape.SetDim(2, pattern_num);
        newshape.SetDim(3, kernel_u * kernel_v);
        newshape.SetPadding1(0, (newshape.GetDim(0) + align_size - 1) / align_size * align_size - newshape.GetDim(0));

        auto status = GetCudaDevice()->Realloc(newshape, &weight_buffer);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
            return status;
        }
        auto stream = GetStream();
        auto size = PPLConvTransposeGetFilterBufSizeCudaFp16(W->GetShape());
        ALLOC_BUFFERF_FOR_ALGO_SELECT(filter_temp_buffer, size, ALGO_MAX_TIME)
        ALLOC_BUFFERF_FOR_ALGO_SELECT(filter_input_buffer, W->GetShape()->CalcBytesIncludingPadding(), ALGO_MAX_TIME)
        auto filter_shape = *W->GetShape(); filter_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
        GetCudaDevice()->GetDataConverter()->Convert(&filter_input_buffer, filter_shape, W->GetBufferDesc(), *W->GetShape());
        PPLCUDAConvTransposeCvt(GetCudaDevice()->GetDeviceProp(), stream, filter_input_buffer.addr, filter_temp_buffer.addr,
                                weight_buffer.addr, W->GetShape(), &param_kernel_);
    }
    ppl::common::Destructor __tmp_buffer_guard__([this, &weight_buffer]() -> void {
        GetCudaDevice()->Free(&weight_buffer);
    });

    if (ctx->GetInputCount() >= 3) {
        B = ctx->GetInput<TensorImpl>(2);
        b_data = B->GetBufferPtr<float>();
    }
    fuse_param_t temp_fuse_param;
    ConvertToForwardFuseParam(ctx, GetCudaDevice(), param_->extra_param.fuse_info, temp_fuse_param);
    CUfunction module_func = nullptr;
#ifdef PPLNN_ENABLE_CUDA_JIT
    CUDAModule* module = static_cast<CUDAModule*>(this->GetCommonParam()->module);
    module_func = module->GetKernelFunc();
#endif

    status = PPLCUDAConvTransposeForward(GetCudaDevice()->GetDeviceProp(), GetStream(), module_func, X->GetShape(), X->GetBufferPtr(),
                                         param_->extra_param.is_initializer_weight ? (int4*)ctx->GetInput<TensorImpl>(1)->GetBufferPtr() : (int4*)weight_buffer.addr,
                                         b_data, Y->GetShape(), Y->GetBufferPtr(), &param_kernel_, param_->extra_param.algo_info,
                                         temp_fuse_param, tmp_buffer);

    return status;
}

}}} // namespace ppl::nn::cuda
