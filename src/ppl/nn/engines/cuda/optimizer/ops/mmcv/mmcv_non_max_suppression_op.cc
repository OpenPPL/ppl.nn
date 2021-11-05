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

#include "ppl/nn/engines/cuda/optimizer/ops/mmcv/mmcv_non_max_suppression_op.h"

#include "ppl/nn/engines/cuda/kernels/mmcv/mmcv_non_max_suppression_kernel.h"
#include "ppl/nn/oputils/mmcv/reshape_mmcv_non_max_suppression.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

RetCode MMCVNonMaxSupressionOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<MMCVNMSParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        auto shape0 = &info->GetInput<TensorImpl>(0)->GetShape();
        shape0->SetDataType(DATATYPE_FLOAT32);

        auto shape1 = &info->GetInput<TensorImpl>(1)->GetShape();
        shape1->SetDataType(DATATYPE_FLOAT32);

        auto shape = &info->GetOutput<TensorImpl>(0)->GetShape();
        shape->SetDataType(DATATYPE_INT64);
        return RC_SUCCESS;
    };

    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeMMCVNonMaxSuppression(info, nullptr);
    };

    return RC_SUCCESS;
}

RetCode MMCVNonMaxSupressionOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* MMCVNonMaxSupressionOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<MMCVNonMaxSuppressionKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
