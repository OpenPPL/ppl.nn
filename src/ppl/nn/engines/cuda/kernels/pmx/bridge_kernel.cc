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

#include "ppl/nn/engines/cuda/kernels/pmx/bridge_kernel.h"

#include "cudakernel/reformat/reformat.h"
#include "ppl/common/cuda/cuda_types.h"
#include "ppl/nn/common/logger.h"

namespace ppl { namespace nn { namespace cuda {

bool BridgeKernel::EqualTypeAndFormat(const TensorImpl* input, const TensorImpl* output, const CudaTensorQuant& in_quant, const CudaTensorQuant& out_quant) {
    auto src_align_size = ppl::common::cuda::GetDataFormatChannelAlignment(input->GetShape()->GetDataFormat());
    auto dst_align_size = ppl::common::cuda::GetDataFormatChannelAlignment(output->GetShape()->GetDataFormat());
    CudaTensorKernelQuant in_quant_kernel, out_quant_kernel;
    in_quant_kernel.format = in_quant.format; in_quant_kernel.type = in_quant.type;
    in_quant_kernel.per_channel = in_quant.per_channel; in_quant_kernel.bit_width = in_quant.bit_width;
    in_quant_kernel.scale = in_quant.scale; in_quant_kernel.zero_point = in_quant.zero_point;
    out_quant_kernel.format = out_quant.format; out_quant_kernel.type = out_quant.type;
    out_quant_kernel.per_channel = out_quant.per_channel; out_quant_kernel.bit_width = out_quant.bit_width;
    out_quant_kernel.scale = out_quant.scale; out_quant_kernel.zero_point = out_quant.zero_point;

    if (input->GetShape()->GetDataType() != output->GetShape()->GetDataType()) {
        return false;
    }

    if (input->GetShape()->GetDataType() == ppl::common::DATATYPE_INT8 && !EqualQuant(in_quant_kernel, out_quant_kernel)) {
        return false;
    }

    if (input->GetShape()->GetDataFormat() == output->GetShape()->GetDataFormat()) {
        return true;
    }

    if (input->GetShape()->GetDimCount() == 1 && output->GetShape()->GetDimCount() == 1) {
        return true;
    }

    if (input->GetShape()->GetDim(1) % src_align_size != 0 || output->GetShape()->GetDim(1) % dst_align_size != 0) {
        return false;
    }

    if (input->GetShape()->GetDimCount() == 2 && output->GetShape()->GetDimCount() == 2) {
        return true;
    }

    if (input->GetShape()->GetDimCount() == 4 && output->GetShape()->GetDimCount() == 4 &&
        input->GetShape()->GetDim(2) == 1 && input->GetShape()->GetDim(3) == 1) {
        return true;
    }

    return false;
}

ppl::common::RetCode BridgeKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    auto converter = output->GetDevice()->GetDataConverter();

    auto input_id = input->GetEdge()->GetId();
    auto input_quant = GetCommonParam()->cuda_tensor_info->at(input_id);
    auto output_id = output->GetEdge()->GetId();
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output_id);

    if (input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL &&
        EqualTypeAndFormat(input, output, input_quant, output_quant)) {
        output->TransferBufferFrom(input);
        return status;
    }

    if (input->GetShape()->GetDataType() != ppl::common::DATATYPE_INT8 &&
        output->GetShape()->GetDataType() != ppl::common::DATATYPE_INT8) {
        status = converter->Convert(&output->GetBufferDesc(), *output->GetShape(), input->GetBufferDesc(), *input->GetShape());
    } else {
        status = ((CudaDataConverter*)converter)->Convert(&output->GetBufferDesc(), *output->GetShape(), output_quant, input->GetBufferDesc(), *input->GetShape(), input_quant);
    }

    return status;
}

}}} // namespace ppl::nn::cuda
