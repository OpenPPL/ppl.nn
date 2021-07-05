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

#include "ppl/nn/engines/x86/kernels/onnx/scatter_nd_kernel.h"
#include "ppl/nn/common/logger.h"

#include "ppl/kernel/x86/fp32/scatter_nd.h"

namespace ppl { namespace nn { namespace x86 {

bool ScatterNdKernel::CanDoExecute(const KernelExecContext& ctx) const {
    return ctx.GetInput<TensorImpl>(0)->GetShape().GetBytesIncludingPadding() != 0;
}

ppl::common::RetCode ScatterNdKernel::DoExecute(KernelExecContext* ctx) {
    auto x = ctx->GetInput<TensorImpl>(0);
    auto indices = ctx->GetInput<TensorImpl>(1);
    auto updates = ctx->GetInput<TensorImpl>(2);
    auto y = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [x]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_X86_DEBUG_TRACE("Input [indices]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(indices);
    PPLNN_X86_DEBUG_TRACE("Input [updates]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(updates);
    PPLNN_X86_DEBUG_TRACE("Output [y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(y);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const uint32_t r = x->GetShape().GetDimCount();
    const uint32_t q = indices->GetShape().GetDimCount();
    const uint32_t k = indices->GetShape().GetDim(q - 1);

    int32_t inner_dim = 1;
    for (uint32_t i = k; i < r; i++) {
        inner_dim *= x->GetShape().GetDim(i);
    }
    int32_t num_indices = 1;
    for (uint32_t i = 0; i < q - 1; i++) {
        num_indices *= indices->GetShape().GetDim(i);
    }
    int32_t indices_dim = k;

    auto dim_count = x->GetShape().GetDimCount();
    std::vector<int32_t> strides_vec(dim_count);
    auto strides = strides_vec.data(); // may be faster

    strides[dim_count - 1] = 1;
    for (int i = dim_count - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * x->GetShape().GetDim(i + 1);
    }
    const auto data_type = x->GetShape().GetDataType();
    const auto data_format = x->GetShape().GetDataFormat();

    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        if (data_type == ppl::common::DATATYPE_FLOAT32) {
            return kernel::x86::scatter_nd_ndarray_fp32(x->GetBufferPtr<float>(), updates->GetBufferPtr<float>(),
                                                        indices->GetBufferPtr<int64_t>(), strides,
                                                        x->GetShape().GetElementsIncludingPadding(), inner_dim,
                                                        num_indices, indices_dim, y->GetBufferPtr<float>());
        } else {
            LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
        }
    } else {
        LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format);
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::x86
