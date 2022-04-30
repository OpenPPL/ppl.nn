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

#include "ppl/nn/engines/arm/kernels/onnx/pad_kernel.h"
#include "ppl/kernel/arm_server/pad/neon/pad.h"
#include "ppl/kernel/arm_server/common/memory.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode PadKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_ARM_REQUIRED_INPUT(x, 0);
    PPLNN_ARM_OPTIONAL_INPUT(pads, 1);
    PPLNN_ARM_OPTIONAL_INPUT(constant, 2);
    PPLNN_ARM_REQUIRED_OUTPUT(y, 0);

    const float param_const_val = param_->value;

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [x]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(x);

    const int dim_count = x->GetShape()->GetDimCount();
    std::vector<int64_t> pads_value(2 * dim_count, 0);
    if (pads) {
        PPLNN_ARM_DEBUG_TRACE("Input [pads]:\n");
        PPL_ARM_TENSOR_PRINT_DEBUG_MSG(pads);
        auto pads_data = pads->GetBufferPtr<int64_t>();
        for (int64_t i = 0; i < 2 * dim_count; ++i) {
            pads_value[i] = pads_data[i];
        }
    } else {
        for (int64_t i = 0; i < 2 * dim_count; ++i) {
            pads_value[i] = (int64_t)param_->pads[i];
        }
    }

    uint64_t cvt_param_const_val = (uint64_t)0;
    void* constant_value = &cvt_param_const_val;
    if (constant) {
        PPLNN_ARM_DEBUG_TRACE("Input [constant]:\n");
        PPL_ARM_TENSOR_PRINT_DEBUG_MSG(constant);
        constant_value = constant->GetBufferPtr<void>();
    } else {
        const auto data_type = x->GetShape()->GetDataType();
        switch (data_type) {
            case ppl::common::DATATYPE_UINT8:
                *((uint8_t*)constant_value) = static_cast<uint8_t>(param_const_val);
                break;
            case ppl::common::DATATYPE_UINT16:
                *((uint16_t*)constant_value) = static_cast<uint16_t>(param_const_val);
                break;
            case ppl::common::DATATYPE_UINT32:
                *((uint32_t*)constant_value) = static_cast<uint32_t>(param_const_val);
                break;
            case ppl::common::DATATYPE_UINT64:
                *((uint64_t*)constant_value) = static_cast<uint64_t>(param_const_val);
                break;

            case ppl::common::DATATYPE_FLOAT16:
                *((__fp16*)constant_value) = static_cast<__fp16>(param_const_val);
                break;
            case ppl::common::DATATYPE_FLOAT32:
                *((float*)constant_value) = static_cast<float>(param_const_val);
                break;
            case ppl::common::DATATYPE_FLOAT64:
                *((double*)constant_value) = static_cast<double>(param_const_val);
                break;

            case ppl::common::DATATYPE_INT8:
                *((int8_t*)constant_value) = static_cast<int8_t>(param_const_val);
                break;
            case ppl::common::DATATYPE_INT16:
                *((int16_t*)constant_value) = static_cast<int16_t>(param_const_val);
                break;
            case ppl::common::DATATYPE_INT32:
                *((int32_t*)constant_value) = static_cast<int32_t>(param_const_val);
                break;
            case ppl::common::DATATYPE_INT64:
                *((int64_t*)constant_value) = static_cast<int64_t>(param_const_val);
                break;

            case ppl::common::DATATYPE_BOOL:
                *((bool*)constant_value) = static_cast<bool>(param_const_val);
                break;

            default:
                LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
                return ppl::common::RC_UNSUPPORTED;
                ;
        }
    }

    PPLNN_ARM_DEBUG_TRACE("Output [y]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(y);
    PPLNN_ARM_DEBUG_TRACE("pad mode: %d\n", param_->mode);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    auto start_pads = pads_value.data();
    auto end_pads = pads_value.data() + dim_count;

    if (x->GetShape()->GetElementsExcludingPadding() ==
        y->GetShape()->GetElementsExcludingPadding()) { // no padding at all, just copy
        if (x->GetEdge()->CalcConsumerCount() == 1 && x->GetType() == TENSORTYPE_NORMAL) {
            y->TransferBufferFrom(x);
        } else {
            ppl::kernel::arm_server::memory_copy(x->GetBufferPtr(), x->GetShape()->GetBytesIncludingPadding(),
                                                 y->GetBufferPtr());
        }
        return ppl::common::RC_SUCCESS;
    }

    switch (param_->mode) {
        case ppl::nn::onnx::PadParam::PAD_MODE_CONSTANT:
            return ppl::kernel::arm_server::neon::pad_constant(x->GetShape(), y->GetShape(), x->GetBufferPtr<void>(),
                                                               start_pads, end_pads, constant_value,
                                                               y->GetBufferPtr<void>());
        case ppl::nn::onnx::PadParam::PAD_MODE_REFLECT:
            return ppl::kernel::arm_server::neon::pad_reflect(x->GetShape(), y->GetShape(), x->GetBufferPtr<void>(),
                                                              start_pads, end_pads, constant_value,
                                                              y->GetBufferPtr<void>());
        case ppl::nn::onnx::PadParam::PAD_MODE_EDGE:
            return ppl::kernel::arm_server::neon::pad_edge(x->GetShape(), y->GetShape(), x->GetBufferPtr<void>(),
                                                           start_pads, end_pads, constant_value,
                                                           y->GetBufferPtr<void>());
        default:
            break;
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::arm
