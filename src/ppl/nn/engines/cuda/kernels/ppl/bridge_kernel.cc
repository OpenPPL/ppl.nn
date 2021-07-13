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
#include "ppl/nn/common/logger.h"

namespace ppl { namespace nn { namespace cuda {

bool BridgeKernel::EqualTypeAndFormat(const TensorImpl* input, const TensorImpl* output) {
    if (input->GetShape().GetDataType() != output->GetShape().GetDataType()) {
        return false;
    }

    if (input->GetShape().GetDataFormat() == output->GetShape().GetDataFormat()) {
        return true;
    }

    if (input->GetShape().GetDimCount() == 2 && output->GetShape().GetDimCount() == 2) {
        return true;
    }

    return false;
}

ppl::common::RetCode BridgeKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;

    auto converter = output->GetDevice()->GetDataConverter();
    status =
        converter->Convert(&output->GetBufferDesc(), output->GetShape(), input->GetBufferDesc(), input->GetShape());
    return status;
}

}}} // namespace ppl::nn::cuda
