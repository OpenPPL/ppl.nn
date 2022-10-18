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

#include "ppl/nn/engines/cuda/kernels/onnx/reciprocal_kernel.h"

#include "cudakernel/unary/unary_zero.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode ReciprocalKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    auto input_quant = GetCommonParam()->cuda_tensor_info->at(input->GetEdge()->GetId());
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output->GetEdge()->GetId());

    if(input->GetShape()->GetDataType() != ppl::common::DATATYPE_FLOAT32 || 
        input->GetShape()->GetDataType() != ppl::common::DATATYPE_FLOAT16){
        LOG(ERROR) << "Reciprocal op only support float16 and float32 for now.";
        return ppl::common::RC_INVALID_VALUE;
    }

    ppl::common::RetCode status = PPLCUDAUnaryZeroReciprocalForwardImp(GetStream(), input->GetShape(), input->GetBufferPtr(),
                                                                output->GetShape(), output->GetBufferPtr());

    return status;
}

}}} // namespace ppl::nn::cuda
