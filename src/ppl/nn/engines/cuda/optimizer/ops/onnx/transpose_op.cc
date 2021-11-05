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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/transpose_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/transpose_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_transpose.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

RetCode TransposeOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<TransposeParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
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
        const TensorShape& in_shape0 = info->GetInput<TensorImpl>(0)->GetShape();

        TransposeParam modified_param = param_;
        if (modified_param.perm.empty()) {
            int32_t dim_count = in_shape0.GetDimCount();
            modified_param.perm.resize(dim_count);
            for (int it = 0; it < dim_count; ++it) {
                modified_param.perm[it] = dim_count - it - 1;
            }
        }

        if (modified_param.perm.size() != in_shape0.GetDimCount()) {
            return RC_UNSUPPORTED;
        }

        return oputils::ReshapeTranspose(info, &modified_param);
    };

    return RC_SUCCESS;
}

RetCode TransposeOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* TransposeOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<TransposeKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
