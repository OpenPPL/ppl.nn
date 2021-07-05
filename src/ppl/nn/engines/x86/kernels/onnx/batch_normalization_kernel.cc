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

#include "ppl/nn/engines/x86/kernels/onnx/batch_normalization_kernel.h"

#include "ppl/kernel/x86/fp32/batchnorm.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode BatchNormalizationKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [input]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_X86_DEBUG_TRACE("Input [output]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_X86_DEBUG_TRACE("epsilon: %lf\n", param_->epsilon);
    PPLNN_X86_DEBUG_TRACE("momentum: %lf\n", param_->momentum);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_format = input->GetShape().GetDataFormat();
    const auto data_type = input->GetShape().GetDataType();

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        if (data_format == ppl::common::DATAFORMAT_N16CX) {
            if (MayUseISA(ppl::common::ISA_X86_AVX)) {
                if (fuse_relu_) {
                    return kernel::x86::batchnorm_n16cx_fp32_avx<true>(
                        &input->GetShape(), input->GetBufferPtr<const float>(),
                        ctx->GetInput<TensorImpl>(3)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(4)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(1)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(2)->GetBufferPtr<float>(), param_->epsilon,
                        output->GetBufferPtr<float>());
                } else {
                    return kernel::x86::batchnorm_n16cx_fp32_avx<false>(
                        &input->GetShape(), input->GetBufferPtr<const float>(),
                        ctx->GetInput<TensorImpl>(3)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(4)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(1)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(2)->GetBufferPtr<float>(), param_->epsilon,
                        output->GetBufferPtr<float>());
                }
            } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
                if (fuse_relu_) {
                    return kernel::x86::batchnorm_n16cx_fp32_sse<true>(
                        &input->GetShape(), input->GetBufferPtr<const float>(),
                        ctx->GetInput<TensorImpl>(3)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(4)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(1)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(2)->GetBufferPtr<float>(), param_->epsilon,
                        output->GetBufferPtr<float>());
                } else {
                    return kernel::x86::batchnorm_n16cx_fp32_sse<false>(
                        &input->GetShape(), input->GetBufferPtr<const float>(),
                        ctx->GetInput<TensorImpl>(3)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(4)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(1)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(2)->GetBufferPtr<float>(), param_->epsilon,
                        output->GetBufferPtr<float>());
                }
            } else {
                LOG(ERROR) << "ISA not supported";
            }
        } else if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
            if (MayUseISA(ppl::common::ISA_X86_AVX)) {
                if (fuse_relu_) {
                    return kernel::x86::batchnorm_ndarray_fp32_avx<true>(
                        &input->GetShape(), input->GetBufferPtr<const float>(),
                        ctx->GetInput<TensorImpl>(3)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(4)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(1)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(2)->GetBufferPtr<float>(), param_->epsilon,
                        output->GetBufferPtr<float>());
                } else {
                    return kernel::x86::batchnorm_ndarray_fp32_avx<false>(
                        &input->GetShape(), input->GetBufferPtr<const float>(),
                        ctx->GetInput<TensorImpl>(3)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(4)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(1)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(2)->GetBufferPtr<float>(), param_->epsilon,
                        output->GetBufferPtr<float>());
                }
            } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
                if (fuse_relu_) {
                    return kernel::x86::batchnorm_ndarray_fp32_sse<true>(
                        &input->GetShape(), input->GetBufferPtr<const float>(),
                        ctx->GetInput<TensorImpl>(3)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(4)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(1)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(2)->GetBufferPtr<float>(), param_->epsilon,
                        output->GetBufferPtr<float>());
                } else {
                    return kernel::x86::batchnorm_ndarray_fp32_sse<false>(
                        &input->GetShape(), input->GetBufferPtr<const float>(),
                        ctx->GetInput<TensorImpl>(3)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(4)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(1)->GetBufferPtr<float>(),
                        ctx->GetInput<TensorImpl>(2)->GetBufferPtr<float>(), param_->epsilon,
                        output->GetBufferPtr<float>());
                }
            } else {
                LOG(ERROR) << "ISA not supported";
            }
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format) << ".";
        }
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::x86
