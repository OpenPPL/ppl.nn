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

#include "ppl/nn/engines/x86/kernels/onnx/transpose_kernel.h"
#include "ppl/nn/common/logger.h"
#include "ppl/kernel/x86/fp32/transpose.h"
#include "ppl/kernel/x86/int64/transpose.h"
#include "ppl/kernel/x86/fp32/reorder.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode TransposeKernel::DoExecute(KernelExecContext* ctx) {
    auto data = ctx->GetInput<TensorImpl>(0);
    auto transposed = ctx->GetOutput<TensorImpl>(0);

    const uint32_t dim_count = data->GetShape().GetDimCount();

    auto modified_perm = param_->perm;
    if (modified_perm.empty()) { // perm is empty, default is reverse dimention.
        modified_perm.resize(dim_count);
        for (size_t i = 0; i < dim_count; ++i) {
            modified_perm[i] = dim_count - i - 1;
        }
    }

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [data]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(data);
    PPLNN_X86_DEBUG_TRACE("Output [transposed]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(transposed);
    for (uint32_t i = 0; i < data->GetShape().GetDimCount(); ++i) {
        PPLNN_X86_DEBUG_TRACE("perm[%u]: %d\n", i, modified_perm[i]);
    }
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = data->GetShape().GetDataType();
    const auto data_format = data->GetShape().GetDataFormat();

    if (data_format == ppl::common::DATAFORMAT_N16CX) {
        if (data_type == ppl::common::DATATYPE_FLOAT32 &&
            transposed->GetShape().GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY &&
            data->GetShape().GetDimCount() == 4 && modified_perm == std::vector<int32_t>{0, 2, 3, 1} &&
            MayUseISA(ppl::common::ISA_X86_AVX) && param_->reverse == false) { // actually N16CHW -> NHWC
            if (MayUseISA(ppl::common::ISA_X86_AVX)) {
                return ppl::kernel::x86::reorder_n16cx_nxc_fp32_avx(&data->GetShape(), data->GetBufferPtr<float>(),
                                                                    transposed->GetBufferPtr<float>());
            } else {
                return ppl::kernel::x86::reorder_n16cx_nxc_fp32(&data->GetShape(), data->GetBufferPtr<float>(),
                                                                transposed->GetBufferPtr<float>());
            }
        }
        LOG(ERROR) << "transpose n16cx only support fp32 4-D tensor input & ndarray output & perm 0,2,3,1 now.";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (dim_count >= 3) {
        std::vector<uint32_t> transpose_dim;
        transpose_dim.reserve(dim_count);
        for (uint32_t i = 0; i < dim_count; i++) {
            if (modified_perm[i] != (int32_t)i) {
                transpose_dim.push_back(i);
            }
        }
        if (transpose_dim.size() == 2 && transpose_dim[0] + 1 == transpose_dim[1] && transpose_dim[1] + 1 < dim_count) {
            if (data_type == ppl::common::DATATYPE_FLOAT32) {
                return ppl::kernel::x86::transpose_ndarray_continous2d_fp32(
                    &data->GetShape(), data->GetBufferPtr<float>(), transpose_dim[0], transpose_dim[1],
                    transposed->GetBufferPtr<float>());
            } else if (data_type == ppl::common::DATATYPE_INT64) {
                return ppl::kernel::x86::transpose_ndarray_continous2d_int64(
                    &data->GetShape(), data->GetBufferPtr<int64_t>(), transpose_dim[0], transpose_dim[1],
                    transposed->GetBufferPtr<int64_t>());
            }
        }
    }

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        return kernel::x86::transpose_ndarray_fp32(&data->GetShape(), &transposed->GetShape(),
                                                   data->GetBufferPtr<const float>(), modified_perm.data(),
                                                   transposed->GetBufferPtr<float>());
    } else if (data_type == ppl::common::DATATYPE_INT64) {
        return kernel::x86::transpose_ndarray_int64(&data->GetShape(), &transposed->GetShape(),
                                                    data->GetBufferPtr<const int64_t>(), modified_perm.data(),
                                                    transposed->GetBufferPtr<int64_t>());
    } else {
        LOG(ERROR) << "unsupported DataType.";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
