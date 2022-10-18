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

#include "ppl/nn/engines/cuda/kernels/onnx/mod_kernel.h"

#include "cudakernel/arithmetic/arithmetic.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode ModKernel::DoExecute(KernelExecContext* ctx) {
    auto input0 = ctx->GetInput<TensorImpl>(0);
    auto input1 = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);
    auto input_id0 = input0->GetEdge()->GetId();
    auto input_id1 = input1->GetEdge()->GetId();
    auto input_quant0 = GetCommonParam()->cuda_tensor_info->at(input_id0);
    auto input_quant1 = GetCommonParam()->cuda_tensor_info->at(input_id1);
    auto output_id = output->GetEdge()->GetId();
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output_id);

    auto fmod = param_->fmod;
    auto in_type0 = input0->GetShape()->GetDataType();
    auto in_type1 = input1->GetShape()->GetDataType();
    
    if(in_type0 != in_type1) {
        LOG(ERROR) << "Input datatype of modkernel should be the same";
        return ppl::common::RC_INVALID_VALUE;
    }
    if(in_type0 != ppl::common::DATATYPE_INT64 || in_type0 != ppl::common::DATATYPE_INT32 
        || in_type0 != ppl::common::DATATYPE_INT16 || in_type0 != ppl::common::DATATYPE_INT8){
            LOG(ERROR) << "Only support integer mod now";
            return ppl::common::RC_INVALID_VALUE;
        }
    if(fmod == 1){
        LOG(ERROR) << "Not support float mod yet";
        return ppl::common::RC_INVALID_VALUE;
    }

    ppl::common::RetCode status =
        PPLCUDAArithMeticModForwardImp(GetStream(), input0->GetShape(), input0->GetBufferPtr(), input1->GetShape(),
                                       input1->GetBufferPtr(), output->GetShape(), output->GetBufferPtr(), 
                                       input_quant0.scale[0], input_quant1.scale[0], output_quant.scale[0]);
    
    return status;
}

}}} // namespace ppl::nn::cuda
