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

#include "online_dequantize_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/pmx/i8i8/online_dequantize_kernel.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;


namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

RetCode I8I8OnlineDequantizeOp::DoInit(const OptKernelOptions& options) {

    infer_type_and_format_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto scale_outer_shape = info->GetInput<TensorImpl>(1)->GetShape();
        auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();

        output_shape->SetDataFormat(DATAFORMAT_NDARRAY);
        output_shape->SetDataType(scale_outer_shape->GetDataType());

        return RC_SUCCESS;
    };
    infer_dims_func_ = GenericInferDims;

    return RC_SUCCESS;
}

KernelImpl* I8I8OnlineDequantizeOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<I8I8OnlineDequantizeKernel>(&param_);
}



}}}}} // namespace ppl::nn::llm::cuda::pmx
