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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/range_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/range_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_range.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode RangeOp::Init(const OptKernelOptions& options) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        auto shape = &info->GetInput<TensorImpl>(0)->GetShape();
        type = shape->GetDataType();
        return InferDefaultType(info, type);
    };

    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        if (info->GetInputCount() != 3 || info->GetOutputCount() != 1) {
            return RC_INVALID_VALUE;
        }
        if (!info->GetInput<TensorImpl>(0)->GetBufferPtr() || !info->GetInput<TensorImpl>(1)->GetBufferPtr() ||
            !info->GetInput<TensorImpl>(2)->GetBufferPtr()) {
            return RC_NOT_FOUND;
        }

        const auto data_type = info->GetInput<TensorImpl>(0)->GetShape().GetDataType();
        if (data_type == DATATYPE_INT64) {
            int64_t start = 0, limit = 0, delta = 1;
            auto status = info->GetInput<TensorImpl>(0)->CopyToHost(&start);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "Copy start failed: " << GetRetCodeStr(status);
                return status;
            }
            status = info->GetInput<TensorImpl>(1)->CopyToHost(&limit);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "Copy limit failed: " << GetRetCodeStr(status);
                return status;
            }
            status = info->GetInput<TensorImpl>(2)->CopyToHost(&delta);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "Copy delta failed: " << GetRetCodeStr(status);
                return status;
            }
            return oputils::ReshapeRange(info, start, limit, delta);
        } else if (data_type == DATATYPE_FLOAT32) {
            float start = 0.0f, limit = 0.0f, delta = 1.0f;
            auto status = info->GetInput<TensorImpl>(0)->CopyToHost(&start);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "Copy start failed: " << GetRetCodeStr(status);
                return status;
            }
            status = info->GetInput<TensorImpl>(1)->CopyToHost(&limit);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "Copy limit failed: " << GetRetCodeStr(status);
                return status;
            }
            status = info->GetInput<TensorImpl>(2)->CopyToHost(&delta);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "Copy delta failed: " << GetRetCodeStr(status);
                return status;
            }
            return oputils::ReshapeRange(info, start, limit, delta);
        } else {
            return RC_UNSUPPORTED;
        }
    };

    return RC_SUCCESS;
}

RetCode RangeOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* RangeOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<RangeKernel>();
}

}}} // namespace ppl::nn::cuda
