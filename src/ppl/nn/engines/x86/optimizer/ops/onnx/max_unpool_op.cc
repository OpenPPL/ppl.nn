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

#include "ppl/nn/engines/x86/optimizer/ops/onnx/max_unpool_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/max_unpool_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_maxunpool.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode MaxUnPoolOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto ret = oputils::ReshapeMaxUnpool(info, param_.get());
        if (ret != RC_SUCCESS) {
            return ret;
        }

        if (info->GetInput<TensorImpl>(0)->GetShape().GetDataType() != DATATYPE_FLOAT32) {
            LOG(ERROR) << "tensor X only support fp32 now.";
            return RC_UNSUPPORTED;
        }
        if (info->GetInput<TensorImpl>(1)->GetShape().GetDataType() != DATATYPE_INT64) {
            LOG(ERROR) << "tensor I only support fp32 now.";
            return RC_UNSUPPORTED;
        }
        if (info->GetInput<TensorImpl>(0)->GetShape().GetDataFormat() != DATAFORMAT_NDARRAY ||
            info->GetInput<TensorImpl>(1)->GetShape().GetDataFormat() != DATAFORMAT_NDARRAY) {
            LOG(ERROR) << "only support ndarray now.";
            return RC_UNSUPPORTED;
        }
        if (info->GetInput<TensorImpl>(0)->GetShape().GetDimCount() != 4 ||
            info->GetInput<TensorImpl>(1)->GetShape().GetDimCount() != 4) {
            LOG(ERROR) << "only support 4-D tensor now.";
            return RC_UNSUPPORTED;
        }
        return RC_SUCCESS;
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

KernelImpl* MaxUnPoolOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<MaxUnpoolKernel>(param_.get());
}

}}} // namespace ppl::nn::x86
