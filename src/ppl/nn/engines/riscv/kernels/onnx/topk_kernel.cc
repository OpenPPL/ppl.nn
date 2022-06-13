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

#include "ppl/nn/engines/riscv/kernels/onnx/topk_kernel.h"
#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/utils/destructor.h"
#include "ppl/nn/common/logger.h"
#include "ppl/kernel/riscv/fp32/topk.h"
#include "ppl/kernel/riscv/fp16/topk.h"

namespace ppl { namespace nn { namespace riscv {

uint64_t TopKKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto input_shape = ctx.GetInput<TensorImpl>(0)->GetShape();
    uint32_t axis = param_->axis < 0 ? param_->axis + input_shape->GetDimCount() : param_->axis;

    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        return ppl::kernel::riscv::topk_ndarray_get_buffer_bytes_fp32(ctx.GetInput<TensorImpl>(0)->GetShape(), axis);
    } else if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        return ppl::kernel::riscv::topk_ndarray_get_buffer_bytes_fp16(ctx.GetInput<TensorImpl>(0)->GetShape(), axis);
    } else {
        return 1;
    }
}

ppl::common::RetCode TopKKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_RISCV_REQUIRED_INPUT(X, 0);
    PPLNN_RISCV_OPTIONAL_INPUT(K, 1);
    PPLNN_RISCV_REQUIRED_OUTPUT(Values, 0);
    PPLNN_RISCV_REQUIRED_OUTPUT(Indices, 1);

    int64_t k_val = *K->GetBufferPtr<int64_t>();

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [X]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(X);
    if (K) {
        PPLNN_RISCV_DEBUG_TRACE("Input [K]:\n");
        PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(K);
        k_val = K->GetBufferPtr<const int64_t>()[0];
    }

    PPLNN_RISCV_DEBUG_TRACE("k: %ld\n", k_val);

    // PPLNN_RISCV_REALLOC_TENSOR_BUFFER(Values);
    PPLNN_RISCV_DEBUG_TRACE("Output [Values]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(Values);
    // PPLNN_RISCV_REALLOC_TENSOR_BUFFER(Indices);
    PPLNN_RISCV_DEBUG_TRACE("Output [Indices]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(Indices);

    if (k_val == -1) {
        LOG(ERROR) << "Get undefined k";
        return ppl::common::RC_UNSUPPORTED;
    }

    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_size = CalcTmpBufferSize(*ctx);
    auto status = GetRiscvDevice()->AllocTmpBuffer(tmp_buffer_size, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_size << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    utils::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
        GetRiscvDevice()->FreeTmpBuffer(&tmp_buffer_desc);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;
    PPLNN_RISCV_DEBUG_TRACE("buffer: %p\n", tmp_buffer);

    uint32_t axis = param_->axis < 0 ? param_->axis + X->GetShape()->GetDimCount() : param_->axis;

    auto data_type = X->GetShape()->GetDataType();
    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        return kernel::riscv::topk_ndarray_fp32(X->GetShape(), Values->GetShape(), Indices->GetShape(),
                                                X->GetBufferPtr<const float>(), k_val, axis, param_->largest,
                                                param_->sorted, tmp_buffer, Values->GetBufferPtr<float>(),
                                                Indices->GetBufferPtr<int64_t>());
    } else if (data_type == ppl::common::DATATYPE_FLOAT16) {
        return kernel::riscv::topk_ndarray_fp16(X->GetShape(), Values->GetShape(), Indices->GetShape(),
                                                X->GetBufferPtr<const __fp16>(), k_val, axis, param_->largest,
                                                param_->sorted, tmp_buffer, Values->GetBufferPtr<__fp16>(),
                                                Indices->GetBufferPtr<int64_t>());

    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::riscv
