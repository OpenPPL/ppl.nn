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

#include "ppl/nn/engines/arm/kernels/onnx/topk_kernel.h"
#include "ppl/nn/utils/destructor.h"
#include "ppl/kernel/arm_server/topk/neon/topk.h"

namespace ppl { namespace nn { namespace arm {

uint64_t TopKKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    uint32_t axis =
        param_->axis < 0 ? param_->axis + ctx.GetInput<TensorImpl>(0)->GetShape()->GetDimCount() : param_->axis;
    return ppl::kernel::arm_server::neon::topk_ndarray_get_buffer_bytes(ctx.GetInput<TensorImpl>(0)->GetShape(), axis);
}


ppl::common::RetCode TopKKernel::DoExecute(KernelExecContext* ctx) {
    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_size = CalcTmpBufferSize(*ctx);
    auto status = GetArmDevice()->AllocTmpBuffer(tmp_buffer_size, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_size << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    utils::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
        GetArmDevice()->FreeTmpBuffer(&tmp_buffer_desc);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;

    PPLNN_ARM_REQUIRED_INPUT(x, 0);
    PPLNN_ARM_OPTIONAL_INPUT(k, 1);
    PPLNN_ARM_REQUIRED_OUTPUT(values, 0);
    PPLNN_ARM_REQUIRED_OUTPUT(indices, 1);

    int64_t k_val = param_->k;

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [x]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(x);
    if (k) {
        PPLNN_ARM_DEBUG_TRACE("Input [k]:\n");
        PPL_ARM_TENSOR_PRINT_DEBUG_MSG(k);
        k_val = k->GetBufferPtr<const int64_t>()[0];
    }
    PPLNN_ARM_DEBUG_TRACE("k: %ld\n", k_val);
    PPLNN_ARM_DEBUG_TRACE("Output [values]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(values);
    PPLNN_ARM_DEBUG_TRACE("Output [indices]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(indices);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    if (k_val == -1) {
        LOG(ERROR) << "Get undefined k";
        return ppl::common::RC_UNSUPPORTED;
    }

    const auto data_type = x->GetShape()->GetDataType();
    const auto data_format = x->GetShape()->GetDataFormat();
    if (data_format != ppl::common::DATAFORMAT_NDARRAY) {
        return ppl::common::RC_UNSUPPORTED;
    }
    if (data_type == ppl::common::DATATYPE_FLOAT16 && !MayUseISA(ppl::common::ISA_ARMV8_2)) {
        LOG(ERROR) << "fp16 needs isa >= armv8.2.";
        return ppl::common::RC_UNSUPPORTED;
    }

    uint32_t axis = param_->axis < 0 ? param_->axis + x->GetShape()->GetDimCount() : param_->axis;

    return ppl::kernel::arm_server::neon::topk(x->GetShape(), values->GetShape(), indices->GetShape(),
                                            x->GetBufferPtr<void>(), k_val, axis, param_->largest,
                                            param_->sorted, tmp_buffer, values->GetBufferPtr<void>(),
                                            indices->GetBufferPtr<int64_t>());

}

}}} // namespace ppl::nn::arm
