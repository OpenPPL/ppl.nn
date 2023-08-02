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

#include "ppl/nn/engines/cuda/optimizer/ops/pmx/rms_norm_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/pmx/rms_norm_kernel.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;


namespace ppl { namespace nn { namespace cuda {

RetCode RMSNormOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<RMSNormParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    return RC_SUCCESS;
}

RMSNormOp::RMSNormOp(const ir::Node* node) : CudaOptKernel(node) {

    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        auto dtype = type;
        if (dtype == DATATYPE_UNKNOWN) {
            dtype = ppl::common::DATATYPE_FLOAT16;
        }
        if (dtype == DATATYPE_INT8) {
            LOG(ERROR) << "currently not support int8/quant, please check the model";
            return ppl::common::RC_INVALID_VALUE;
        }

        for (uint32_t i = 0; i < info->GetInputCount(); ++i) {
            auto input = info->GetInput<TensorImpl>(i);
            auto in_shape = input->GetShape();
            in_shape->SetDataType(dtype);
        }

        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto output = info->GetOutput<TensorImpl>(i);
            auto out_shape = output->GetShape();
            out_shape->SetDataType(dtype);
        }

        return ppl::common::RC_SUCCESS;
    };

    infer_dims_func_ = GenericInferDims;

}

RetCode RMSNormOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* RMSNormOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<RMSNormKernel>(&param_);
}



}}} // namespace ppl::nn::cuda
