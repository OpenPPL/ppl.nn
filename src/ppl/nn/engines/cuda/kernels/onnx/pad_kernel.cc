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

#include "ppl/nn/engines/cuda/kernels/onnx/pad_kernel.h"

#include <memory>

#include "cudakernel/memory/pad.h"
#include "ppl/common/destructor.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode PadKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    PadKernelParam kernel_param;

    void* pad_buffer = ctx->GetInput<TensorImpl>(1)->GetBufferPtr();
    // keep pads data on host for optimization
    uint32_t num_elems = ctx->GetInput<TensorImpl>(1)->GetShape()->CalcElementsIncludingPadding();
    std::vector<int64_t> pads;
    pads.resize(num_elems);
    ctx->GetInput<TensorImpl>(1)->CopyToHost(pads.data());
    kernel_param.pads.resize(num_elems);
    for (uint32_t i = 0; i < num_elems; ++i) {
        kernel_param.pads[i] = pads[i];
    }

    kernel_param.mode = param_->mode;

    auto constant_data = ctx->GetInput<TensorImpl>(2);
    auto status = constant_data->CopyToHost(&(kernel_param.constant_value));
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "Copy constant value failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }

    if (input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL &&
        input->GetShape()->CalcElementsIncludingPadding() == output->GetShape()->CalcElementsIncludingPadding()) {
        output->TransferBufferFrom(input);
    } else {
        status = PPLCUDAPadForwardImp(GetStream(), kernel_param, input->GetShape(), input->GetBufferPtr(),
                                      (const int64_t*)pad_buffer, output->GetShape(), output->GetBufferPtr());
    }
    return status;
}

}}} // namespace ppl::nn::cuda
