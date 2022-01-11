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

#include "ppl/nn/engines/arm/kernels/onnx/batch_normalization_kernel.h"
#include "ppl/kernel/arm_server/batchnorm/neon/batchnorm.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode BatchNormalizationKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_ARM_REQUIRED_INPUT(X, 0);
    PPLNN_ARM_REQUIRED_INPUT(scale, 1);
    PPLNN_ARM_REQUIRED_INPUT(B, 2);
    PPLNN_ARM_REQUIRED_INPUT(mean, 3);
    PPLNN_ARM_REQUIRED_INPUT(var, 4);
    PPLNN_ARM_REQUIRED_OUTPUT(Y, 0);

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [X]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(X);
    PPLNN_ARM_DEBUG_TRACE("Input [scale]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(scale);
    PPLNN_ARM_DEBUG_TRACE("Input [B]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(B);
    PPLNN_ARM_DEBUG_TRACE("Input [mean]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(mean);
    PPLNN_ARM_DEBUG_TRACE("Input [var]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(var);
    PPLNN_ARM_DEBUG_TRACE("Input [Y]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(Y);
    PPLNN_ARM_DEBUG_TRACE("epsilon: %lf\n", param_->epsilon);
    PPLNN_ARM_DEBUG_TRACE("momentum: %lf\n", param_->momentum);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    return ppl::kernel::arm_server::neon::batchnorm(
        X->GetShape(), X->GetBufferPtr<void>(), mean->GetBufferPtr<void>(), var->GetBufferPtr<void>(),
        scale->GetBufferPtr<void>(), B->GetBufferPtr<void>(), param_->epsilon, fuse_relu_, Y->GetBufferPtr<void>());
}

}}} // namespace ppl::nn::arm
