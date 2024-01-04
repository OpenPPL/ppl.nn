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

#include "reshape_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/onnx/reshape_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_reshape.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;


namespace ppl { namespace nn { namespace llm { namespace cuda { namespace onnx {

RetCode ReshapeOp::CommonInit() {
    infer_type_and_format_func_ = GenericInferTypeAndFormat;
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        if (constant_shape_data_.empty()) {
            auto shape = info->GetInput<TensorImpl>(1);
            if (!shape->GetBufferPtr()) {
                return RC_NOT_FOUND;
            }

            vector<int64_t> shape_data(shape->GetShape()->CalcElementsIncludingPadding());
            auto status = shape->CopyToHost(shape_data.data());
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "shape->CopyToHost() failed: " << GetRetCodeStr(status);
                return status;
            }

            return ppl::nn::onnx::ReshapeReshape(info, param_.get(), shape_data.data());
        } else {
            return ppl::nn::onnx::ReshapeReshape(info, param_.get(), constant_shape_data_.data());
        }
    };

    return RC_SUCCESS;
}

RetCode ReshapeOp::DoInit(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ppl::nn::onnx::ReshapeParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenericLoadParam failed: " << GetRetCodeStr(status);
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
    } else {
        LOG(WARNING) << "non-constant reshape will cause performance downgrade.";
    }

    return CommonInit();
}

KernelImpl* ReshapeOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ReshapeKernel>();
}

}}}}} // namespace ppl::nn::llm::cuda::pmx
