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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/reciprocal_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/reciprocal_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode ReciprocalOp::Init(const OptKernelOptions& options) {
    return RC_SUCCESS;
}

ReciprocalOp::ReciprocalOp(const ir::Node* node) : CudaOptKernel(node) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status;
        if (type == DATATYPE_UNKNOWN) {
            status = InferInheritedType(info);
        } else if (type == DATATYPE_INT8) {
            // reciprocal only support float type
            for (uint32_t i = 0; i < info->GetInputCount(); ++i) {
                auto shape = info->GetInput<TensorImpl>(i)->GetShape();
                shape->SetDataType(ppl::common::DATATYPE_FLOAT32);
            }
            info->GetOutput<TensorImpl>(0)->GetShape()->SetDataType(ppl::common::DATATYPE_FLOAT32);
        } else {
            status = InferDefaultType(info, type);
        }
        return status;
    };

    infer_dims_func_ = GenericInferDims;
}

RetCode ReciprocalOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* ReciprocalOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ReciprocalKernel>();
}

}}} // namespace ppl::nn::cuda
