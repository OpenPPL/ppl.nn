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

#include "ppl/nn/engines/x86/kernels/onnx/reduce_sum_kernel.h"
#include "ppl/nn/common/logger.h"

#include "ppl/kernel/x86/fp32/reduce.h"
#include "ppl/kernel/x86/int64/reduce.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode ReduceSumKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(data, 0);
    PPLNN_X86_REQUIRED_OUTPUT(reduced, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [data]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(data);

    const uint32_t dim_count = data->GetShape()->GetDimCount();
    auto fixed_axes = param_->axes;
    if (param_->axes.empty()) { // empty axes means reduce all dims
        fixed_axes.resize(dim_count);
        for (size_t i = 0; i < dim_count; i++) {
            fixed_axes[i] = i;
        }
    }

    for (uint32_t i = 0; i < fixed_axes.size(); i++) {
        if (fixed_axes[i] < 0) { // turn negative axes to positive axes
            fixed_axes[i] = fixed_axes[i] + dim_count;
        }
    }
    for (uint32_t i = 0; i < fixed_axes.size(); ++i) {
        PPLNN_X86_DEBUG_TRACE("axes[%d]: %d\n", i, fixed_axes[i]);
    }
    PPLNN_X86_DEBUG_TRACE("keepdims: %d\n", param_->keep_dims);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    PPLNN_X86_REALLOC_TENSOR_BUFFER(reduced);
    PPLNN_X86_DEBUG_TRACE("Output [reduced]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(reduced);

    auto data_type = data->GetShape()->GetDataType();
    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        if (MayUseISA(ppl::common::ISA_X86_AVX)) {
            return kernel::x86::reduce_sum_fp32_avx(data->GetShape(), reduced->GetShape(),
                                                    data->GetBufferPtr<float>(), fixed_axes.data(), fixed_axes.size(),
                                                    reduced->GetBufferPtr<float>());
        } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {
            return kernel::x86::reduce_sum_fp32_sse(data->GetShape(), reduced->GetShape(),
                                                    data->GetBufferPtr<float>(), fixed_axes.data(), fixed_axes.size(),
                                                    reduced->GetBufferPtr<float>());
        } else {
            LOG(ERROR) << "get unsupported isa " << GetISA();
            return ppl::common::RC_UNSUPPORTED;
        }
    } else if (data_type == ppl::common::DATATYPE_INT64) {
        return kernel::x86::reduce_sum_int64(data->GetShape(), reduced->GetShape(), data->GetBufferPtr<int64_t>(),
                                             fixed_axes.data(), fixed_axes.size(), reduced->GetBufferPtr<int64_t>());
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
