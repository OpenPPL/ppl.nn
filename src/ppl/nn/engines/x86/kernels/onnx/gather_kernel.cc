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

#include "ppl/nn/engines/x86/kernels/onnx/gather_kernel.h"
#include "ppl/common/destructor.h"
#include "ppl/nn/common/logger.h"

#include "ppl/kernel/x86/fp32/gather.h"
#include "ppl/kernel/x86/int64/gather.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode GatherKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(x, 0);
    PPLNN_X86_REQUIRED_INPUT(indices, 1);
    PPLNN_X86_REQUIRED_OUTPUT(y, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [x]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_X86_DEBUG_TRACE("Input [indices]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(indices);

    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    PPLNN_X86_REALLOC_TENSOR_BUFFER(y);
    PPLNN_X86_DEBUG_TRACE("Output [y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(y);

    const int64_t r = x->GetShape()->GetDimCount();
    const int64_t q = indices->GetShape()->GetDimCount();
    const int64_t real_axis = param_->axis >= 0 ? param_->axis : param_->axis + r;

    int64_t num_indices = 1;
    const int64_t indices_dim = indices->GetShape()->GetDim(q - 1);
    int64_t outter_dim = 1;
    int64_t inner_dim = 1;
    const int64_t gather_dim = x->GetShape()->GetDim(real_axis);

    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_size = CalcTmpBufferSize(*ctx);
    auto status = GetX86Device()->AllocTmpBuffer(indices->GetShape()->CalcBytesExcludingPadding(), &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_size << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    ppl::common::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
        GetX86Device()->FreeTmpBuffer(&tmp_buffer_desc);
    });
    auto real_indices = (int64_t*)tmp_buffer_desc.addr;
    auto indices_data = indices->GetBufferPtr<const int64_t>();
    for (uint64_t i = 0; i < indices->GetShape()->CalcElementsExcludingPadding(); ++i) {
        real_indices[i] = indices_data[i] >= 0 ? indices_data[i] : indices_data[i] + gather_dim;
    }

    for (int64_t i = 0; i < q - 1; ++i) {
        num_indices *= indices->GetShape()->GetDim(i);
    }
    for (int64_t i = 0; i < real_axis; ++i) {
        outter_dim *= x->GetShape()->GetDim(i);
    }
    for (int64_t i = real_axis + 1; i < r; ++i) {
        inner_dim *= x->GetShape()->GetDim(i);
    }

    const ppl::common::datatype_t data_type = y->GetShape()->GetDataType();
    const auto data_format = x->GetShape()->GetDataFormat();
    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        if (data_type == ppl::common::DATATYPE_FLOAT32) {
            return kernel::x86::gather_ndarray_fp32(x->GetBufferPtr<const float>(), real_indices, outter_dim,
                                                    gather_dim, inner_dim, num_indices, indices_dim,
                                                    y->GetBufferPtr<float>());
        } else if (data_type == ppl::common::DATATYPE_INT64) {
            return kernel::x86::gather_ndarray_int64(x->GetBufferPtr<const int64_t>(), real_indices, outter_dim,
                                                     gather_dim, inner_dim, num_indices, indices_dim,
                                                     y->GetBufferPtr<int64_t>());
        } else {
            LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type);
        }
    } else {
        LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
