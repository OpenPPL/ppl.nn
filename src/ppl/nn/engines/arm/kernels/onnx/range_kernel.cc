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

#include "ppl/nn/engines/arm/kernels/onnx/range_kernel.h"
#include "ppl/nn/engines/arm/utils/macros.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode RangeKernel::DoExecute(KernelExecContext* ctx) {
    auto start = ctx->GetInput<TensorImpl>(0);
    auto limit = ctx->GetInput<TensorImpl>(1);
    auto delta = ctx->GetInput<TensorImpl>(2);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [start]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(start);
    PPLNN_ARM_DEBUG_TRACE("Input [limit]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(limit);
    PPLNN_ARM_DEBUG_TRACE("Input [delta]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(delta);
    PPLNN_ARM_DEBUG_TRACE("Output [output]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = output->GetShape()->GetDataType();
    if (data_type == ppl::common::DATATYPE_INT64) {
        const int64_t start_val = start->GetBufferPtr<int64_t>()[0];
        const int64_t delta_val = delta->GetBufferPtr<int64_t>()[0];
        int64_t* output_data = output->GetBufferPtr<int64_t>();
        for (uint32_t i = 0; i < output->GetShape()->CalcElementsExcludingPadding(); i++) {
            output_data[i] = start_val + i * delta_val;
        }
    } else if (data_type == ppl::common::DATATYPE_FLOAT32) {
        const float start_val = start->GetBufferPtr<float>()[0];
        const float delta_val = delta->GetBufferPtr<float>()[0];
        float* output_data = output->GetBufferPtr<float>();
        for (uint32_t i = 0; i < output->GetShape()->CalcElementsExcludingPadding(); i++) {
            output_data[i] = start_val + i * delta_val;
        }
#ifdef PPLNN_USE_ARMV8_2_FP16
    } else if (data_type == ppl::common::DATATYPE_FLOAT16) {
        const __fp16 start_val = start->GetBufferPtr<__fp16>()[0];
        const __fp16 delta_val = delta->GetBufferPtr<__fp16>()[0];
        __fp16* output_data = output->GetBufferPtr<__fp16>();
        for (uint32_t i = 0; i < output->GetShape()->CalcElementsExcludingPadding(); i++) {
            output_data[i] = start_val + i * delta_val;
        }
#endif
    } else {
        LOG(ERROR) << "unsupported data type: " << common::GetDataTypeStr(data_type) << ".";
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::arm
