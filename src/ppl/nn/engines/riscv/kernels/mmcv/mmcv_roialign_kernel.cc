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

#include "ppl/nn/engines/riscv/kernels/mmcv/mmcv_roialign_kernel.h"

#include "ppl/kernel/riscv/fp32/mmcv_roialign.h"

namespace ppl { namespace nn { namespace riscv {

ppl::common::RetCode MMCVROIAlignKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto rois = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());

    PPLNN_RISCV_DEBUG_TRACE("Input [input]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_RISCV_DEBUG_TRACE("Input [rois]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(rois);

    PPLNN_RISCV_REALLOC_TENSOR_BUFFER(output);
    PPLNN_RISCV_DEBUG_TRACE("Output [output]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(output);

    auto data_type = input->GetShape()->GetDataType();
    auto data_format = input->GetShape()->GetDataFormat();
    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
            return kernel::riscv::mmcv_roialign_ndarray_fp32(
                input->GetShape(), rois->GetShape(), output->GetShape(), input->GetBufferPtr<const float>(),
                rois->GetBufferPtr<const float>(), param_->aligned, param_->sampling_ratio, param_->spatial_scale,
                param_->pool_mode == "avg", output->GetBufferPtr<float>());
        } else {
            LOG(ERROR) << "unsupported data format: " << ppl::common::GetDataFormatStr(data_format);
        }
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type);
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::riscv