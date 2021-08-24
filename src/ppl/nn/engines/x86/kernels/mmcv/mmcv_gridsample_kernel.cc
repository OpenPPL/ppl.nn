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

#include "ppl/nn/engines/x86/kernels/mmcv/mmcv_gridsample_kernel.h"

#include "ppl/kernel/x86/fp32/mmcv_gridsample.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode MMCVGridSampleKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto grid = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [input]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_X86_DEBUG_TRACE("Input [grid]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(grid);
    PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_X86_DEBUG_TRACE("align_corners: %ld\n", param_->align_corners);
    PPLNN_X86_DEBUG_TRACE("interpolation_mode: %ld\n", param_->interpolation_mode);
    PPLNN_X86_DEBUG_TRACE("padding_mode: %ld\n", param_->padding_mode);

    if (param_->interpolation_mode == 0) { // bilinear
        if (false) {
        }
#ifdef PPL_USE_X86_AVX512
        else if (MayUseISA(ppl::common::ISA_X86_AVX512)) {
            return kernel::x86::mmcv_gridsample_linear_ndarray_fp32_avx512(
                &input->GetShape(), &grid->GetShape(), input->GetBufferPtr<float>(), grid->GetBufferPtr<float>(),
                param_->align_corners, param_->padding_mode, output->GetBufferPtr<float>());
        }
#endif
        else {
            return kernel::x86::mmcv_gridsample_linear_ndarray_fp32(
                &input->GetShape(), &grid->GetShape(), input->GetBufferPtr<float>(), grid->GetBufferPtr<float>(),
                param_->align_corners, param_->padding_mode, output->GetBufferPtr<float>());
        }
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
