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

#include "ppl/nn/engines/arm/kernels/onnx/leaky_relu_kernel.h"
#include "ppl/kernel/arm_server/leaky_relu/neon/leaky_relu.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode LeakyReLUKernel::DoExecute(KernelExecContext* ctx) {
    auto x = ctx->GetInput<TensorImpl>(0);
    auto y = ctx->GetOutput<TensorImpl>(0);

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [x]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_ARM_DEBUG_TRACE("Output [y]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(y);
    PPLNN_ARM_DEBUG_TRACE("alpha: %f\n", param_->alpha);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = x->GetShape()->GetDataType();
    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        if (MayUseISA(ppl::common::ISA_ARMV8)) {
            return ppl::kernel::arm_server::neon::leaky_relu_fp32(x->GetShape(), x->GetBufferPtr<float>(),
                                                                  param_->alpha, y->GetBufferPtr<float>());
        }
    } 
#ifdef PPLNN_USE_ARMV8_2_FP16
    else if (data_type == ppl::common::DATATYPE_FLOAT16) {
        if (MayUseISA(ppl::common::ISA_ARMV8_2)) {
            return ppl::kernel::arm_server::neon::leaky_relu_fp16(x->GetShape(), x->GetBufferPtr<__fp16>(),
                                                                  param_->alpha, y->GetBufferPtr<__fp16>());
        }
    } 
#endif
    else {
        LOG(ERROR) << "unsupported datatype: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::arm
