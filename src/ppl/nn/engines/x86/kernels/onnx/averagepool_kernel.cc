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

#include "ppl/nn/engines/x86/kernels/onnx/averagepool_kernel.h"
#include "ppl/kernel/x86/fp32/averagepool2d.h"
#include "ppl/common/destructor.h"

namespace ppl { namespace nn { namespace x86 {

uint64_t AveragePoolKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto src = ctx.GetInput<TensorImpl>(0);
    auto dst = ctx.GetOutput<TensorImpl>(0);
    int64_t pad_w = 0;
    if (param_->global_pooling == 0 && param_->pads.size() >= 2) {
        pad_w = param_->pads[1];
    }

    const auto data_type = src->GetShape()->GetDataType();
    const auto data_format = src->GetShape()->GetDataFormat();
    if (data_format != ppl::common::DATAFORMAT_NDARRAY)
        return 64u;
    if (data_type != ppl::common::DATATYPE_FLOAT32)
        return 64u;
    if (MayUseISA(ppl::common::ISA_X86_SSE))
        return kernel::x86::averagepool_fp32_get_buffer_bytes(src->GetShape(), dst->GetShape(), pad_w);
    return 64u;
}

ppl::common::RetCode AveragePoolKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(X, 0);
    PPLNN_X86_REQUIRED_OUTPUT(Y, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());

    PPLNN_X86_DEBUG_TRACE("Input [X]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(X);

    PPLNN_X86_DEBUG_TRACE("mode: %d\n", param_->mode);
    PPLNN_X86_DEBUG_TRACE("ceil_mode: %d\n", param_->ceil_mode);
    PPLNN_X86_DEBUG_TRACE("global_pooling: %d\n", param_->global_pooling);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    PPLNN_X86_REALLOC_TENSOR_BUFFER(Y);
    PPLNN_X86_DEBUG_TRACE("Output [Y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);

    if (X->GetShape()->GetDimCount() != 4) {
        LOG(ERROR) << "only support 4-D tensor now.";
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t src_h = X->GetShape()->GetDim(2);
    const int64_t src_w = X->GetShape()->GetDim(3);

    int64_t kernel_h;
    int64_t kernel_w;
    int64_t stride_h;
    int64_t stride_w;
    int64_t pad_h;
    int64_t pad_w;
    int64_t dilation_h;
    int64_t dilation_w;
    if (param_->global_pooling) {
        kernel_h = src_h;
        kernel_w = src_w;
        stride_h = src_h;
        stride_w = src_w;
        pad_h = 0;
        pad_w = 0;
        dilation_h = 1;
        dilation_w = 1;
    } else {
        kernel_h = param_->kernel_shape[0];
        kernel_w = param_->kernel_shape[1];
        stride_h = param_->strides.size() >= 1 ? param_->strides[0] : 1;
        stride_w = param_->strides.size() >= 2 ? param_->strides[1] : 1;
        pad_h = param_->pads.size() >= 1 ? param_->pads[0] : 0;
        pad_w = param_->pads.size() >= 2 ? param_->pads[1] : 0;
        if ((param_->pads.size() >= 3 && param_->pads[2] != pad_h) ||
            (param_->pads.size() >= 4 && param_->pads[3] != pad_w)) {
            LOG(ERROR) << "only support symmetrical pads now.";
            return ppl::common::RC_UNSUPPORTED;
        }
        dilation_h = param_->dilations.size() >= 1 ? param_->dilations[0] : 1;
        dilation_w = param_->dilations.size() >= 2 ? param_->dilations[1] : 1;
        if (dilation_h != 1 || dilation_w != 1) {
            LOG(ERROR) << "only support dilation = 1 now.";
            return ppl::common::RC_UNSUPPORTED;
        }
    }

    PPLNN_X86_DEBUG_TRACE("kernel_shape: %ld %ld\n", kernel_h, kernel_w);
    PPLNN_X86_DEBUG_TRACE("dilations: %ld %ld\n", dilation_h, dilation_w);
    PPLNN_X86_DEBUG_TRACE("strides: %ld %ld\n", stride_h, stride_w);
    PPLNN_X86_DEBUG_TRACE("pads: %ld %ld\n", pad_h, pad_w);

    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_size = CalcTmpBufferSize(*ctx);
    auto status = GetX86Device()->AllocTmpBuffer(tmp_buffer_size, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_size << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    ppl::common::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
        GetX86Device()->FreeTmpBuffer(&tmp_buffer_desc);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;
    PPLNN_X86_DEBUG_TRACE("buffer: %p\n", tmp_buffer);

    const auto data_type = X->GetShape()->GetDataType();
    const auto data_format = X->GetShape()->GetDataFormat();

    if (data_format == ppl::common::DATAFORMAT_N16CX) {
        if (data_type == ppl::common::DATATYPE_FLOAT32) {
            if (false) {
            }
#ifdef PPL_USE_X86_AVX512
            else if (MayUseISA(ppl::common::ISA_X86_AVX512)) {
                return ppl::kernel::x86::averagepool2d_n16cx_blk1x16_fp32_avx512(
                    X->GetShape(), Y->GetShape(), X->GetBufferPtr<float>(), kernel_h, kernel_w, stride_h, stride_w,
                    pad_h, pad_w, param_->mode, param_->ceil_mode, Y->GetBufferPtr<float>());
            }
#endif
            else if (MayUseISA(ppl::common::ISA_X86_AVX)) {
                return ppl::kernel::x86::averagepool2d_n16cx_blk1x8_fp32_avx(
                    X->GetShape(), Y->GetShape(), X->GetBufferPtr<float>(), kernel_h, kernel_w, stride_h, stride_w,
                    pad_h, pad_w, param_->mode, param_->ceil_mode, Y->GetBufferPtr<float>());
            } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
                return ppl::kernel::x86::averagepool2d_n16cx_blk1x4_fp32_sse(
                    X->GetShape(), Y->GetShape(), X->GetBufferPtr<float>(), kernel_h, kernel_w, stride_h, stride_w,
                    pad_h, pad_w, param_->mode, param_->ceil_mode, Y->GetBufferPtr<float>());
            } else {
                LOG(ERROR) << "get unsupported isa " << GetISA() << ".";
            }
        } else {
            LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
        }
    } else if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        if (data_type == ppl::common::DATATYPE_FLOAT32) {
            if (MayUseISA(ppl::common::ISA_X86_SSE)) {
                return ppl::kernel::x86::averagepool2d_ndarray_normal_fp32_sse(
                    X->GetShape(), Y->GetShape(), X->GetBufferPtr<float>(), kernel_h, kernel_w, stride_h, stride_w, pad_h,
                    pad_w, param_->mode, param_->ceil_mode, tmp_buffer, Y->GetBufferPtr<float>());
            } else {
                return ppl::kernel::x86::averagepool2d_ndarray_normal_fp32(
                X->GetShape(), Y->GetShape(), X->GetBufferPtr<float>(), kernel_h, kernel_w, stride_h, stride_w,
                pad_h, pad_w, param_->mode, param_->ceil_mode, Y->GetBufferPtr<float>());
            }
        } else {
            LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
        }
    } else {
        LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
