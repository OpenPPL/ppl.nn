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

#include "ppl/nn/engines/riscv/kernels/pmx/reorder_kernel.h"
#include "ppl/kernel/riscv/common/math.h"
#include "ppl/nn/common/logger.h"
#include "ppl/common/types.h"
#include <string.h>
#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/engines/riscv/utils/data_trans.h"
#include "ppl/nn/engines/riscv/utils/fp16fp32_cvt.h"
#include "ppl/nn/runtime/tensor_impl.h"

namespace ppl { namespace nn { namespace riscv {
ppl::common::RetCode ReorderKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [input]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_RISCV_DEBUG_TRACE("Output [output]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(output);

    const ppl::common::datatype_t input_type = input->GetShape()->GetDataType();
    const ppl::common::datatype_t output_type = output->GetShape()->GetDataType();
    const ppl::common::dataformat_t input_format = input->GetShape()->GetDataFormat();
    const ppl::common::dataformat_t output_format = output->GetShape()->GetDataFormat();

    LOG(DEBUG) << "reorder from data format " << ppl::common::GetDataFormatStr(input_format) << " to "
               << ppl::common::GetDataFormatStr(output_format);
    LOG(DEBUG) << "reorder from data type " << ppl::common::GetDataTypeStr(input_type) << " to "
               << ppl::common::GetDataTypeStr(output_type);

    int64_t input_n, input_c, input_h, input_w;
    if (input->GetShape()->GetDimCount() == 2) {
        input_n = input->GetShape()->GetDim(0);
        input_c = input->GetShape()->GetDim(1);
        input_h = 1;
        input_w = 1;
    } else if (input->GetShape()->GetDimCount() == 4) {
        input_n = input->GetShape()->GetDim(0);
        input_c = input->GetShape()->GetDim(1);
        input_h = input->GetShape()->GetDim(2);
        input_w = input->GetShape()->GetDim(3);
    } else if (input_format != output_format) {
        return ppl::common::RC_UNSUPPORTED;
    }

    if (output_format == input_format && input_type == output_type) {
        memcpy(output->GetBufferPtr<__fp16>(), input->GetBufferPtr<__fp16>(),
               input->GetShape()->CalcBytesIncludingPadding());
        return ppl::common::RC_SUCCESS;
    } else {
        if (input_type == ppl::common::DATATYPE_FLOAT32 && output_type == ppl::common::DATATYPE_FLOAT16 &&
            output_format == input_format) {
            int64_t data_cnt = input->GetShape()->CalcElementsIncludingPadding();
            CvtFp32ToFp16(data_cnt, input->GetBufferPtr<float>(), output->GetBufferPtr<__fp16>());
            return ppl::common::RC_SUCCESS;

        } else if (input_type == ppl::common::DATATYPE_FLOAT16 && output_type == ppl::common::DATATYPE_FLOAT32 &&
                   output_format == input_format) {
            int64_t data_cnt = input->GetShape()->CalcElementsIncludingPadding();
            CvtFp16ToFp32(data_cnt, input->GetBufferPtr<__fp16>(), output->GetBufferPtr<float>());
            return ppl::common::RC_SUCCESS;

        } else if (input_type == ppl::common::DATATYPE_FLOAT16 && output_type == ppl::common::DATATYPE_FLOAT32 &&
                   input_format == ppl::common::DATAFORMAT_N8CX && output_format == ppl::common::DATAFORMAT_N4CX) {
            LOG(DEBUG) << "Reorder fp16 n8cx to fp32 n4cx";
            N8cxFp16ToN4cxFp32((input->GetBufferPtr<__fp16>()), input_n, input_c, input_h, input_w,
                              output->GetBufferPtr<float>());
            return ppl::common::RC_SUCCESS;

        } else if (input_type == ppl::common::DATATYPE_FLOAT32 && output_type == ppl::common::DATATYPE_FLOAT16 &&
                   input_format == ppl::common::DATAFORMAT_N4CX && output_format == ppl::common::DATAFORMAT_N8CX) {
            LOG(DEBUG) << "Reorder fp32 n4cx to fp16 n8cx";
            N4cxFp32ToN8cxFp16((input->GetBufferPtr<float>()), input_n, input_c, input_h, input_w,
                              output->GetBufferPtr<__fp16>());
            return ppl::common::RC_SUCCESS;

        } else if (input_type == ppl::common::DATATYPE_FLOAT32 && output_type == ppl::common::DATATYPE_FLOAT16 &&
                   input_format == ppl::common::DATAFORMAT_N4CX && output_format == ppl::common::DATAFORMAT_NDARRAY) {
            LOG(DEBUG) << "Reorder fp32 n4cx to fp16 nadrray";
            N4cxFp32ToNdarrayFp16((input->GetBufferPtr<float>()), input_n, input_c, input_h, input_w,
                              output->GetBufferPtr<__fp16>());
            return ppl::common::RC_SUCCESS;

        } else if (input_type == ppl::common::DATATYPE_FLOAT16 && output_type == input_type &&
                   input_format == ppl::common::DATAFORMAT_NDARRAY && output_format == ppl::common::DATAFORMAT_N8CX) {
            LOG(DEBUG) << "Reorder fp16 ndarray to fp16 n8cx";
            NdarrayToN8cxFp16((input->GetBufferPtr<__fp16>()), input_n, input_c, input_h, input_w,
                              output->GetBufferPtr<__fp16>());
            return ppl::common::RC_SUCCESS;

        } else if (input_type == ppl::common::DATATYPE_FLOAT16 && output_type == input_type &&
                   input_format == ppl::common::DATAFORMAT_N8CX && output_format == ppl::common::DATAFORMAT_NDARRAY) {
            LOG(DEBUG) << "Reorder fp16 n8cx to fp16 ndarray";
            N8cxToNdarrayFp16((input->GetBufferPtr<__fp16>()), input_n, input_c, input_h, input_w,
                              output->GetBufferPtr<__fp16>());
            return ppl::common::RC_SUCCESS;

        } else if (input_type == ppl::common::DATATYPE_FLOAT32 && output_type == input_type &&
                   input_format == ppl::common::DATAFORMAT_NDARRAY && output_format == ppl::common::DATAFORMAT_N8CX) {
            LOG(DEBUG) << "Reorder fp16 ndarray to fp16 n8cx";
            NdarrayToN8cxFp32((input->GetBufferPtr<float>()), input_n, input_c, input_h, input_w,
                              output->GetBufferPtr<float>());
            return ppl::common::RC_SUCCESS;

        } else if (input_type == ppl::common::DATATYPE_FLOAT32 && output_type == input_type &&
                   input_format == ppl::common::DATAFORMAT_N8CX && output_format == ppl::common::DATAFORMAT_NDARRAY) {
            LOG(DEBUG) << "Reorder fp16 n8cx to fp16 ndarray";
            N8cxToNdarrayFp32((input->GetBufferPtr<float>()), input_n, input_c, input_h, input_w,
                              output->GetBufferPtr<float>());
            return ppl::common::RC_SUCCESS;

        } else if (input_type == ppl::common::DATATYPE_FLOAT32 && output_type == input_type &&
                   input_format == ppl::common::DATAFORMAT_NDARRAY && output_format == ppl::common::DATAFORMAT_N4CX) {
            LOG(DEBUG) << "Reorder fp32 ndarray to fp32 n4cx";
            NdarrayToN4cxFp32((input->GetBufferPtr<float>()), input_n, input_c, input_h, input_w,
                              output->GetBufferPtr<float>());
            return ppl::common::RC_SUCCESS;

        } else if (input_type == ppl::common::DATATYPE_FLOAT32 && output_type == input_type &&
                   input_format == ppl::common::DATAFORMAT_N4CX && output_format == ppl::common::DATAFORMAT_NDARRAY) {
            LOG(DEBUG) << "Reorder fp32 n4cx to fp32 ndarray";
            N4cxToNdarrayFp32((input->GetBufferPtr<float>()), input_n, input_c, input_h, input_w,
                              output->GetBufferPtr<float>());
            return ppl::common::RC_SUCCESS;
        } else if (input_type == ppl::common::DATATYPE_FLOAT16 && output_type == ppl::common::DATATYPE_FLOAT32 &&
                   input_format == ppl::common::DATAFORMAT_N8CX && output_format == ppl::common::DATAFORMAT_NDARRAY) {
            LOG(DEBUG) << "Reorder fp16 n8cx to fp32 ndarray";
            N8cxFp16ToNdarrayFp32((input->GetBufferPtr<__fp16>()), input_n, input_c, input_h, input_w,
                                  output->GetBufferPtr<float>());
            return ppl::common::RC_SUCCESS;

        } else if (input_type == ppl::common::DATATYPE_FLOAT32 && output_type == ppl::common::DATATYPE_FLOAT16 &&
                   input_format == ppl::common::DATAFORMAT_NDARRAY && output_format == ppl::common::DATAFORMAT_N8CX) {
            LOG(DEBUG) << "Reorder fp32 ndarray to fp16 n8cx";
            NdarrayFp32ToN8cxFp16((input->GetBufferPtr<float>()), input_n, input_c, input_h, input_w,
                                  output->GetBufferPtr<__fp16>());
            return ppl::common::RC_SUCCESS;

        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
    }
    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::riscv
