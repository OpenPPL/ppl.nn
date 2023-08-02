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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/reshape_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/reshape_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_reshape.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode ReshapeOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    auto node = GetNode();
    auto graph_data = options.graph->data;

    auto shape_data_it = graph_data->constants.find(node->GetInput(1));
    const int64_t* shape_data = nullptr;
    if (shape_data_it != graph_data->constants.end()) {
        shape_data = (const int64_t*)shape_data_it->second.data.GetData();
    }

    if (shape_data != nullptr) {
        auto shape_shape_it = graph_data->shapes.find(node->GetInput(1));
        if (shape_shape_it != graph_data->shapes.end()) {
            auto& shape_shape = shape_shape_it->second;
            constant_shape_data_.assign(shape_data, shape_data + shape_shape.dims[0]);
        }
    }

    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status = RC_SUCCESS;
        if (type == DATATYPE_INT8) {
            status = UnifyToOutputQuant(info, quant);
        } else {
            TensorShape& in_shape = *info->GetInput<TensorImpl>(0)->GetShape();
            if (in_shape.GetDataType() == ppl::common::DATATYPE_UNKNOWN) {
                LOG(ERROR) << "Input edge has unknown type.";
                return RC_INVALID_VALUE;
            }
            for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
                TensorShape& out_shape = *info->GetOutput<TensorImpl>(i)->GetShape();
                out_shape.SetDataType(in_shape.GetDataType());
            }
        }
        auto shape = info->GetInput<TensorImpl>(1)->GetShape();
        shape->SetDataType(DATATYPE_INT64);
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        if (constant_shape_data_.empty()) {
            if (info->GetInputCount() != 2) {
                LOG(ERROR) << "2 input required.";
                return RC_INVALID_VALUE;
            }

            auto input = info->GetInput<TensorImpl>(1);
            if (!input->GetBufferPtr()) {
                return RC_NOT_FOUND;
            }

            const TensorShape& dst_desc = *input->GetShape();
            vector<int64_t> shape_data(dst_desc.CalcElementsIncludingPadding());
            auto status = input->CopyToHost(shape_data.data());
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "Copy shape data failed: " << GetRetCodeStr(status);
                return status;
            }

            return onnx::ReshapeReshape(info, &param_, shape_data.data());
        } else {
            return onnx::ReshapeReshape(info, &param_, constant_shape_data_.data());
        }
    };

    return RC_SUCCESS;
}

RetCode ReshapeOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* ReshapeOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ReshapeKernel>();
}

}}} // namespace ppl::nn::cuda
