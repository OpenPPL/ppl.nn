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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/constant_of_shape_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/constant_of_shape_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_constant_of_shape.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

namespace ppl { namespace nn { namespace cuda {

static RetCode LoadParam(const ir::Node* node, const OptKernelOptions& options, ConstantOfShapeParam* param) {
    auto graph_data = options.graph->data.get();
    auto param_ref = graph_data->attrs.find(node->GetId());
    if (param_ref == graph_data->attrs.end()) {
        return RC_NOT_FOUND;
    }
    auto src_param = static_cast<const ConstantOfShapeParam*>(param_ref->second.get());

    auto rc = param->data.Init(src_param->data.GetSize());
    if (rc != RC_SUCCESS) {
        return rc;
    }

    memcpy(param->data.GetData(), src_param->data.GetData(), src_param->data.GetSize());
    param->data_type = src_param->data_type;
    param->dims = src_param->dims;
    return RC_SUCCESS;
}

RetCode ConstantOfShapeOp::Init(const OptKernelOptions& options) {
    auto status = LoadParam(GetNode(), options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_type_func_ = [this](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        auto output = info->GetOutput<TensorImpl>(0)->GetShape();
        output->SetDataType(param_.data_type);
        return RC_SUCCESS;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        if (info->GetInputCount() != 1 || info->GetOutputCount() != 1) {
            return RC_INVALID_VALUE;
        }

        auto input = info->GetInput<TensorImpl>(0);
        if (!input->GetBufferPtr()) {
            return RC_NOT_FOUND;
        }

        std::vector<int64_t> input_host(input->GetShape()->CalcElementsIncludingPadding());
        auto status = input->CopyToHost(input_host.data());
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Copy input host failed: " << GetRetCodeStr(status);
            return status;
        }

        return onnx::ReshapeConstantOfShape(info, &param_, input_host.data());
    };

    return RC_SUCCESS;
}

RetCode ConstantOfShapeOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* ConstantOfShapeOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ConstantOfShapeKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
