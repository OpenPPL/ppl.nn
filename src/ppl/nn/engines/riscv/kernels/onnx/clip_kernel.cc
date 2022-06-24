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

#include "ppl/nn/engines/riscv/kernels/onnx/clip_kernel.h"
#include "ppl/kernel/riscv/fp16/clip.h"
#include "ppl/kernel/riscv/fp32/clip.h"
#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"
#include <algorithm>
#include <float.h>

namespace ppl { namespace nn { namespace riscv {

bool ClipKernel::CanDoExecute(const KernelExecContext& ctx) const {
    auto tensor = ctx.GetInput<TensorImpl>(0);
    if (!tensor || tensor->GetShape()->CalcBytesIncludingPadding() == 0) {
        return false;
    }
    return true;
}

ppl::common::RetCode ClipKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_RISCV_REQUIRED_INPUT(input, 0);
    PPLNN_RISCV_OPTIONAL_INPUT(min_tensor, 1);
    PPLNN_RISCV_OPTIONAL_INPUT(max_tensor, 2);
    PPLNN_RISCV_REQUIRED_OUTPUT(output, 0);

    const auto input_data_type = input->GetShape()->GetDataType();
    const auto output_data_type = output->GetShape()->GetDataType();
    if (input_data_type != output_data_type) {
        return ppl::common::RC_UNSUPPORTED;
    }

    __fp16 min_val = -FLT_MAX;
    __fp16 max_val = FLT_MAX;
    if (min_tensor) {
        min_val = (__fp16)(min_tensor->GetBufferPtr<float>())[0];
    }
    if (max_tensor) {
        max_val = (__fp16)(max_tensor->GetBufferPtr<float>())[0];
    }
    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [input]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(input);
    if (min_tensor) {
        PPLNN_RISCV_DEBUG_TRACE("Input [min]:\n");
        PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(min_tensor);
    }
    if (max_tensor) {
        PPLNN_RISCV_DEBUG_TRACE("Input [max]:\n");
        PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(max_tensor);
    }
    PPLNN_RISCV_DEBUG_TRACE("Output [output]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_RISCV_DEBUG_TRACE("min_val: %f\n", min_val);
    PPLNN_RISCV_DEBUG_TRACE("max_val: %f\n", max_val);

    if (input_data_type == ppl::common::DATATYPE_FLOAT16) {
        return kernel::riscv::clip_fp16(input->GetShape(), max_val, min_val, input->GetBufferPtr<__fp16>(),
                                        output->GetBufferPtr<__fp16>());
    } else if (input_data_type == ppl::common::DATATYPE_FLOAT32) {
        return kernel::riscv::clip_fp32(input->GetShape(), max_val, min_val, input->GetBufferPtr<float>(),
                                        output->GetBufferPtr<float>());
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(input_data_type);
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::riscv
