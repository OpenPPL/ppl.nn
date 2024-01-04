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

#include "split_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/onnx/split_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_split.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;


namespace ppl { namespace nn { namespace llm { namespace cuda { namespace onnx {

RetCode SplitOp::CommonInit() {
    infer_type_and_format_func_ = GenericInferTypeAndFormat;
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        if (constant_split_data_.empty()) {
            auto split = info->GetInput<TensorImpl>(1);
            if (!split->GetBufferPtr()) {
                return RC_NOT_FOUND;
            }

            vector<int64_t> split_data(split->GetShape()->CalcElementsIncludingPadding());
            auto status = split->CopyToHost(split_data.data());
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "split->CopyToHost() failed: " << GetRetCodeStr(status);
                return status;
            }

            return ppl::nn::onnx::ReshapeSplit(info, param_.get(), split_data.data());
        } else {
            return ppl::nn::onnx::ReshapeSplit(info, param_.get(), constant_split_data_.data());
        }
    };
    return RC_SUCCESS;
}

RetCode SplitOp::DoInit(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ppl::nn::onnx::SplitParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenericLoadParam failed: " << GetRetCodeStr(status);
        return status;
    }

    auto node = GetNode();
    auto graph_data = options.graph->data;
    auto split_data_it = graph_data->constants.find(node->GetInput(1));
    const int64_t* split_data = nullptr;
    if (split_data_it != graph_data->constants.end()) {
        split_data = (const int64_t*)split_data_it->second.data.GetData();
    }
    if (split_data != nullptr) {
        auto split_shape_it = graph_data->shapes.find(node->GetInput(1));
        if (split_shape_it != graph_data->shapes.end()) {
            auto& split_shape = split_shape_it->second;
            constant_split_data_.assign(split_data, split_data + split_shape.dims[0]);
        }
    } else {
        LOG(WARNING) << "non-constant split will cause performance downgrade.";
    }

    return CommonInit();
}

KernelImpl* SplitOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<SplitKernel>(param_.get());
}

}}}}} // namespace ppl::nn::llm::cuda::pmx
