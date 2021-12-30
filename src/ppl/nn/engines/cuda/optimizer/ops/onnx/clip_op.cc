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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/clip_op.h"

#include "ppl/nn/engines/cuda/kernels/onnx/clip_kernel.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode ClipOp::Init(const OptKernelOptions& options) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status;
        type = DATATYPE_FLOAT16;
        if (type == DATATYPE_UNKNOWN) {
            status = InferInheritedType(info);
        } else if (type == DATATYPE_INT8) {
            status = UnifyToOutputQuant(info, quant);
        } else {
            status = InferDefaultType(info, type);
        }
        auto input1 = info->GetInput<TensorImpl>(1);
        if (input1 != nullptr)
            input1->GetShape().SetDataType(DATATYPE_FLOAT32);
        auto input2 = info->GetInput<TensorImpl>(2);
        if (input2 != nullptr)
            input2->GetShape().SetDataType(DATATYPE_FLOAT32);
        return status;
    };

    infer_dims_func_ = GenericInferDims;

    auto data = options.graph->data.get();
    auto min_edge_id = GetNode()->GetInput(1);
    if (min_edge_id != INVALID_EDGEID) {
        param_.min_val = *((float*)data->constants[min_edge_id].data.data());
    }
    auto max_edge_id = GetNode()->GetInput(2);
    if (max_edge_id != INVALID_EDGEID) {
        param_.max_val = *((float*)data->constants[max_edge_id].data.data());
    }
    return RC_SUCCESS;
}

RetCode ClipOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* ClipOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ClipKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
