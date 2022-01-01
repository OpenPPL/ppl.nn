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

#include "ppl/nn/engines/x86/kernels/ppl/reorder_kernel.h"
#include "ppl/nn/common/logger.h"

#include "ppl/kernel/x86/fp32/reorder.h"
#include "ppl/kernel/x86/int64/reorder.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode ReorderKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(input, 0);
    PPLNN_X86_REQUIRED_OUTPUT(output, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());

    PPLNN_X86_DEBUG_TRACE("Input [input]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(input);

    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const ppl::common::datatype_t data_type = input->GetShape()->GetDataType();
    const ppl::common::dataformat_t input_format = input->GetShape()->GetDataFormat();
    const ppl::common::dataformat_t output_format = output->GetShape()->GetDataFormat();

    const bool may_inplace = input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL;

    if (ppl::common::GetSizeOfDataType(data_type) == 4) {
        if (input_format == ppl::common::DATAFORMAT_NDARRAY && output_format == ppl::common::DATAFORMAT_N16CX) {
            const TensorShape padded_input_shape = PadShapeTo3Dims(*input->GetShape());
            if (may_inplace && ppl::kernel::x86::reorder_ndarray_n16cx_may_inplace(&padded_input_shape)) {
                output->TransferBufferFrom(input);
                PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
                PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
                if (MayUseISA(ppl::common::ISA_X86_AVX)) {
                    return ppl::kernel::x86::reorder_ndarray_n16cx_inplace_fp32_avx(
                        &padded_input_shape, output->GetBufferPtr<float>());
                } else {
                    return ppl::kernel::x86::reorder_ndarray_n16cx_inplace_fp32(
                        &padded_input_shape, output->GetBufferPtr<float>());
                }
            } else {
                PPLNN_X86_REALLOC_TENSOR_BUFFER(output);
                PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
                PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
                if (MayUseISA(ppl::common::ISA_X86_AVX)) {
                    return ppl::kernel::x86::reorder_ndarray_n16cx_fp32_avx(
                        &padded_input_shape, input->GetBufferPtr<float>(), output->GetBufferPtr<float>());
                } else {
                    return ppl::kernel::x86::reorder_ndarray_n16cx_fp32(
                        &padded_input_shape, input->GetBufferPtr<float>(), output->GetBufferPtr<float>());
                }
            }
        } else if (input_format == ppl::common::DATAFORMAT_N16CX && output_format == ppl::common::DATAFORMAT_NDARRAY) {
            if (may_inplace && ppl::kernel::x86::reorder_n16cx_ndarray_may_inplace(input->GetShape())) {
                output->TransferBufferFrom(input);
                PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
                PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
                if (MayUseISA(ppl::common::ISA_X86_AVX)) {
                    return ppl::kernel::x86::reorder_n16cx_ndarray_inplace_fp32_avx(
                        input->GetShape(), output->GetBufferPtr<float>());
                } else {
                    return ppl::kernel::x86::reorder_n16cx_ndarray_inplace_fp32(
                        input->GetShape(), output->GetBufferPtr<float>());
                }
            } else {
                PPLNN_X86_REALLOC_TENSOR_BUFFER(output);
                PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
                PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
                if (MayUseISA(ppl::common::ISA_X86_AVX)) {
                    return ppl::kernel::x86::reorder_n16cx_ndarray_fp32_avx(
                        input->GetShape(), input->GetBufferPtr<float>(), output->GetBufferPtr<float>());
                } else {
                    return ppl::kernel::x86::reorder_n16cx_ndarray_fp32(input->GetShape(), input->GetBufferPtr<float>(),
                                                                        output->GetBufferPtr<float>());
                }
            }
        } else {
            LOG(ERROR) << "unsupported reorder from " << ppl::common::GetDataFormatStr(input_format) << " to "
                       << ppl::common::GetDataFormatStr(output_format) << ".";
        }
    } else if (ppl::common::GetSizeOfDataType(data_type) == 8) {
        PPLNN_X86_REALLOC_TENSOR_BUFFER(output);
        PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
        PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
        if (input_format == ppl::common::DATAFORMAT_NDARRAY && output_format == ppl::common::DATAFORMAT_N16CX) {
            const TensorShape padded_input_shape = PadShapeTo3Dims(*input->GetShape());
            if (MayUseISA(ppl::common::ISA_X86_AVX)) {
                return ppl::kernel::x86::reorder_ndarray_n16cx_int64_avx(
                    &padded_input_shape, input->GetBufferPtr<int64_t>(), output->GetBufferPtr<int64_t>());
            } else {
                return ppl::kernel::x86::reorder_ndarray_n16cx_int64(
                    &padded_input_shape, input->GetBufferPtr<int64_t>(), output->GetBufferPtr<int64_t>());
            }
        } else if (input_format == ppl::common::DATAFORMAT_N16CX && output_format == ppl::common::DATAFORMAT_NDARRAY) {
            if (MayUseISA(ppl::common::ISA_X86_AVX)) {
                return ppl::kernel::x86::reorder_n16cx_ndarray_int64_avx(
                    input->GetShape(), input->GetBufferPtr<int64_t>(), output->GetBufferPtr<int64_t>());
            } else {
                return ppl::kernel::x86::reorder_n16cx_ndarray_int64(input->GetShape(), input->GetBufferPtr<int64_t>(),
                                                                     output->GetBufferPtr<int64_t>());
            }
        } else {
            LOG(ERROR) << "unsupported reorder from " << ppl::common::GetDataFormatStr(input_format) << " to "
                       << ppl::common::GetDataFormatStr(output_format) << ".";
        }
    } else {
        LOG(ERROR) << "unsupported data type " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
