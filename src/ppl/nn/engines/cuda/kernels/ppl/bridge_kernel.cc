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

#include "ppl/nn/engines/cuda/kernels/ppl/bridge_kernel.h"

#include "ppl/common/cuda/cuda_types.h"
#include "ppl/nn/common/logger.h"

namespace ppl { namespace nn { namespace cuda {

bool BridgeKernel::EqualTypeAndFormat(const TensorImpl* input, const TensorImpl* output) {
    auto in_shape = input->GetShape();
    auto out_shape = output->GetShape();
    
    if (in_shape.GetDataType() != out_shape.GetDataType()) {
        return false;
    }

    if (in_shape.GetDataFormat() == out_shape.GetDataFormat()) {
        return true;
    }

    auto src_align_size = ppl::common::cuda::GetDataFormatChannelAlignment(in_shape.GetDataFormat());
    auto dst_align_size = ppl::common::cuda::GetDataFormatChannelAlignment(out_shape.GetDataFormat());
    if (in_shape.GetDimCount() == 2 && out_shape.GetDimCount() == 2 &&
        in_shape.GetDim(1) % src_align_size == 0 && out_shape.GetDim(1) % dst_align_size == 0) {
            return true;
    }

    return false;
}

ppl::common::RetCode BridgeKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;

    if (input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL && EqualTypeAndFormat(input, output)) {
        output->TransferBufferFrom(input);
    } else {
        auto converter = output->GetDevice()->GetDataConverter();
        status =
            converter->Convert(&output->GetBufferDesc(), output->GetShape(), input->GetBufferDesc(), input->GetShape());
    }
    return status;
}

}}} // namespace ppl::nn::cuda
