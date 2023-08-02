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

#include "ppl/nn/engines/cuda/kernels/onnx/reshape_kernel.h"

#include "cudakernel/memory/reshape.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode ReshapeKernel::BeforeExecute(KernelExecContext* ctx) {
    auto status = Reshape(ctx);
    if (status != ppl::common::RC_SUCCESS) {
        return status;
    }

    auto input = ctx->GetInput<TensorImpl>(0);
    can_trans_ = ctx->IsLastConsumerOfInput(0) && input->GetType() == TENSORTYPE_NORMAL;

    if (!can_trans_) {
        for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
            auto tensor = ctx->GetOutput<TensorImpl>(i);
            tensor->SetDevice(GetCudaDevice());
            status = tensor->ReallocBuffer();
            if (status != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "ReallocBuffer for tensor[" << tensor->GetName()
                           << "] failed: " << ppl::common::GetRetCodeStr(status);
                return status;
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode ReshapeKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;

    if (can_trans_) {
        output->TransferBufferFrom(input);
    } else {
        status = PPLCUDAReshapeForwardImp(GetStream(), input->GetShape(), input->GetBufferPtr(), output->GetShape(),
                                          output->GetBufferPtr());
    }

    return status;
}

}}} // namespace ppl::nn::cuda
