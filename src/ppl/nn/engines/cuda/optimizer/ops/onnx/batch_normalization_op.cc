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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/batch_normalization_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/batch_normalization_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_batch_normalization.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

namespace ppl { namespace nn { namespace cuda {

RetCode BatchNormalizationOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<BatchNormalizationParam>(options, &param_.param);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        auto& in_shape = *info->GetInput<TensorImpl>(0)->GetShape();
        type = in_shape.GetDataType();
        ppl::common::RetCode status;
        if (type == DATATYPE_UNKNOWN) {
            status = InferInheritedType(info);
        } else if (type == DATATYPE_INT8) {
            status = CopyQuantType(info, quant);
            for (uint32_t i = 1; i < 5; ++i) {
                if (info->GetInputCount() > i) {
                    auto shape = info->GetInput<TensorImpl>(i)->GetShape();
                    shape->SetDataType(ppl::common::DATATYPE_FLOAT32);
                }
            }
        } else {
            status = InferDefaultType(info, type);
        }
        return status;
    };

    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeBatchNormalization(info, nullptr);
    };

    return RC_SUCCESS;
}

RetCode BatchNormalizationOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* BatchNormalizationOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<BatchNormalizationKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
