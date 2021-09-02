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
    PPLNN_X86_REQUIRED_INPUT(X, 0);
    PPLNN_X86_REQUIRED_INPUT(scale, 1);
    PPLNN_X86_REQUIRED_INPUT(B, 2);
    PPLNN_X86_REQUIRED_INPUT(mean, 3);
    PPLNN_X86_REQUIRED_INPUT(var, 4);
    PPLNN_X86_REQUIRED_OUTPUT(Y, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [X]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(X);
    PPLNN_X86_DEBUG_TRACE("Input [scale]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(scale);
    PPLNN_X86_DEBUG_TRACE("Input [B]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(B);
    PPLNN_X86_DEBUG_TRACE("Input [mean]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(mean);
    PPLNN_X86_DEBUG_TRACE("Input [var]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(var);
    PPLNN_X86_DEBUG_TRACE("Input [Y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(Y);
    PPLNN_X86_DEBUG_TRACE("epsilon: %lf\n", param_->epsilon);
    PPLNN_X86_DEBUG_TRACE("momentum: %lf\n", param_->momentum);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_format = X->GetShape().GetDataFormat();
    const auto data_type = X->GetShape().GetDataType();

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        if (data_format == ppl::common::DATAFORMAT_N16CX) {
            if (MayUseISA(ppl::common::ISA_X86_AVX)) {
                return kernel::x86::batchnorm_n16cx_fp32_avx(
                    &X->GetShape(), X->GetBufferPtr<const float>(),
                    mean->GetBufferPtr<float>(), var->GetBufferPtr<float>(),
                    scale->GetBufferPtr<float>(), B->GetBufferPtr<float>(),
                    param_->epsilon, this->fuse_relu_,
                    Y->GetBufferPtr<float>());
            } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
                return kernel::x86::batchnorm_n16cx_fp32_sse(
                    &X->GetShape(), X->GetBufferPtr<const float>(),
                    mean->GetBufferPtr<float>(), var->GetBufferPtr<float>(),
                    scale->GetBufferPtr<float>(), B->GetBufferPtr<float>(),
                    param_->epsilon, this->fuse_relu_,
                    Y->GetBufferPtr<float>());
            } else {
                LOG(ERROR) << "ISA not supported";
            }
        } else if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
            if (MayUseISA(ppl::common::ISA_X86_AVX)) {
                return kernel::x86::batchnorm_ndarray_fp32_avx(
                    &X->GetShape(), X->GetBufferPtr<const float>(),
                    mean->GetBufferPtr<float>(), var->GetBufferPtr<float>(),
                    scale->GetBufferPtr<float>(), B->GetBufferPtr<float>(),
                    param_->epsilon, this->fuse_relu_,
                    Y->GetBufferPtr<float>());
            } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
                return kernel::x86::batchnorm_ndarray_fp32_avx(
                    &X->GetShape(), X->GetBufferPtr<const float>(),
                    mean->GetBufferPtr<float>(), var->GetBufferPtr<float>(),
                    scale->GetBufferPtr<float>(), B->GetBufferPtr<float>(),
                    param_->epsilon, this->fuse_relu_,
                    Y->GetBufferPtr<float>());
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
