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

#include "ppl/nn/engines/arm/kernels/pmx/reorder_kernel.h"
#include "ppl/nn/common/logger.h"
#include "ppl/common/types.h"
#include <arm_neon.h>
#include <string.h>
#include "ppl/nn/engines/arm/utils/macros.h"
#include "ppl/kernel/arm_server/common/data_trans.h"
#include "ppl/kernel/arm_server/common/memory.h"
#include "ppl/kernel/arm_server/cast/neon/cast.h"
#include <stdio.h>
using namespace ppl::kernel::arm_server;

namespace ppl { namespace nn { namespace arm {
ppl::common::RetCode ReorderKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [input]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_ARM_DEBUG_TRACE("Output [output]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    const ppl::common::datatype_t input_type = input->GetShape()->GetDataType();
    const ppl::common::datatype_t output_type = output->GetShape()->GetDataType();
    const ppl::common::dataformat_t input_format = input->GetShape()->GetDataFormat();
    const ppl::common::dataformat_t output_format = output->GetShape()->GetDataFormat();

    if (output_format == input_format && input_type == output_type) {
        return memory_copy(input->GetBufferPtr<char>(), input->GetShape()->GetBytesIncludingPadding(),
                           output->GetBufferPtr<char>());
    } else if (output_format != input_format) {
        if (input_type == ppl::common::DATATYPE_FLOAT32 && output_type == input_type) {
            LOG(DEBUG) << "Reorder between fp32 ndarray and n4cx";
            if (output_format == ppl::common::DATAFORMAT_N4CX && input_format == ppl::common::DATAFORMAT_NDARRAY) {
                NdarrayToN4cxFp32((input->GetBufferPtr<float>()), input->GetShape()->GetDim(0),
                                  input->GetShape()->GetDim(1), input->GetShape()->GetDim(2), input->GetShape()->GetDim(3),
                                  output->GetBufferPtr<float>());
                return ppl::common::RC_SUCCESS;
            } else if (output_format == ppl::common::DATAFORMAT_NDARRAY &&
                       input_format == ppl::common::DATAFORMAT_N4CX) {
                N4cxToNdarrayFp32((input->GetBufferPtr<float>()), input->GetShape()->GetDim(0),
                                  input->GetShape()->GetDim(1), input->GetShape()->GetDim(2), input->GetShape()->GetDim(3),
                                  output->GetBufferPtr<float>());
                return ppl::common::RC_SUCCESS;
            }
        } else if (input_type == ppl::common::DATATYPE_FLOAT16 && output_type == input_type) {
            LOG(DEBUG) << "Reorder between fp16 ndarray and n8cx";
            if (output_format == ppl::common::DATAFORMAT_N8CX && input_format == ppl::common::DATAFORMAT_NDARRAY) {
                NdarrayToN8cxFp16((input->GetBufferPtr<__fp16>()), input->GetShape()->GetDim(0),
                                  input->GetShape()->GetDim(1), input->GetShape()->GetDim(2), input->GetShape()->GetDim(3),
                                  output->GetBufferPtr<__fp16>());
                return ppl::common::RC_SUCCESS;
            } else if (output_format == ppl::common::DATAFORMAT_NDARRAY &&
                       input_format == ppl::common::DATAFORMAT_N8CX) {
                N8cxToNdarrayFp16((input->GetBufferPtr<__fp16>()), input->GetShape()->GetDim(0),
                                  input->GetShape()->GetDim(1), input->GetShape()->GetDim(2), input->GetShape()->GetDim(3),
                                  output->GetBufferPtr<__fp16>());
                return ppl::common::RC_SUCCESS;
            }
        } else if (input_type == ppl::common::DATATYPE_FLOAT16 && output_type == ppl::common::DATATYPE_FLOAT32 &&
                   input_format == ppl::common::DATAFORMAT_N8CX && output_format == ppl::common::DATAFORMAT_NDARRAY) {
            LOG(DEBUG) << "Reorder fp16 n8cx to fp32 ndarray";
            N8cxFp16ToNdarrayFp32((input->GetBufferPtr<__fp16>()), input->GetShape()->GetDim(0),
                                  input->GetShape()->GetDim(1), input->GetShape()->GetDim(2), input->GetShape()->GetDim(3),
                                  output->GetBufferPtr<float>());
            return ppl::common::RC_SUCCESS;
        } else if (input_type == ppl::common::DATATYPE_FLOAT32 && output_type == ppl::common::DATATYPE_FLOAT16 &&
                   input_format == ppl::common::DATAFORMAT_NDARRAY && output_format == ppl::common::DATAFORMAT_N8CX) {
            LOG(DEBUG) << "Reorder fp32 ndarray to fp16 n8cx";
            NdarrayFp32ToN8cxFp16((input->GetBufferPtr<float>()), input->GetShape()->GetDim(0),
                                  input->GetShape()->GetDim(1), input->GetShape()->GetDim(2), input->GetShape()->GetDim(3),
                                  output->GetBufferPtr<__fp16>());
            return ppl::common::RC_SUCCESS;
        }
    } else {
        if (input_type == ppl::common::DATATYPE_FLOAT16 && output_type == ppl::common::DATATYPE_FLOAT32) {
            Fp16ToFp32(input->GetBufferPtr<__fp16>(), input->GetShape()->GetElementsIncludingPadding(),
                       output->GetBufferPtr<float>());
            return ppl::common::RC_SUCCESS;
        } else if (input_type == ppl::common::DATATYPE_FLOAT32 && output_type == ppl::common::DATATYPE_FLOAT16) {
            Fp32ToFp16(input->GetBufferPtr<float>(), input->GetShape()->GetElementsIncludingPadding(),
                       output->GetBufferPtr<__fp16>());
            return ppl::common::RC_SUCCESS;
        } else {
            return ppl::kernel::arm_server::neon::cast(input->GetShape(), output->GetShape(),
                                                       input->GetBufferPtr<void>(), output->GetBufferPtr<void>());
        }
    }
    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::arm
