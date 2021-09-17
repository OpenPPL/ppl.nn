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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/expand_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/expand_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_expand.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode ExpandOp::Init(const OptKernelOptions& options) {
    infer_type_func_ = [this](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status;
        if (type == DATATYPE_UNKNOWN) {
            status = InferInheritedType(info);
        } else if (type == DATATYPE_INT8) {
            status = CopyQuantType(info, quant);
        } else {
            status = InferDefaultType(info, type);
        }
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        if (info->GetInputCount() != 2 || info->GetOutputCount() != 1) {
            return RC_INVALID_VALUE;
        }

        auto shape = info->GetInput<TensorImpl>(1);
        if (!shape->GetBufferPtr()) {
            return RC_NOT_FOUND;
        }

        std::unique_ptr<int64_t[]> shape_ptr(new int64_t[shape->GetShape().GetElementsIncludingPadding()]);
        auto status = shape->CopyToHost(shape_ptr.get());
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Copy shape failed: " << GetRetCodeStr(status);
            return status;
        }

        return oputils::ReshapeExpand(info, nullptr, shape_ptr.get());
    };

    return RC_SUCCESS;
}

RetCode ExpandOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* ExpandOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ExpandKernel>();
}

}}} // namespace ppl::nn::cuda
