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

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/llm_cuda/pmx/generated/llm_cuda_op_params_generated.h"
#endif

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

#ifdef PPLNN_ENABLE_PMX_MODEL
ppl::common::RetCode SplitOp::SerializeData(const ppl::nn::pmx::SerializationContext&, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder builder;
    auto fb_split_point = builder.CreateVector(param_.get()->split_point);
    auto fb_constant_split_data = builder.CreateVector(constant_split_data_);
    auto fb_param = pmx::CreateSplitParam(builder, 
        param_.get()->axis, 
        fb_split_point, 
        fb_constant_split_data);
    auto fb_op_param = pmx::CreateOpParam(builder, pmx::OpParamType_SplitParam, fb_param.Union());
    pmx::FinishOpParamBuffer(builder, fb_op_param);
    return ds->Write(builder.GetBufferPointer(), builder.GetSize());
}

ppl::common::RetCode SplitOp::DeserializeData(const ppl::nn::pmx::DeserializationContext&, const void* base, uint64_t size) {
    auto fb_op_param = pmx::GetOpParam(base);
    auto fb_param = fb_op_param->value_as_SplitParam();
    param_ = make_shared<ppl::nn::onnx::SplitParam>();
    param_.get()->axis = fb_param->axis();
    ppl::nn::pmx::utils::Fbvec2Stdvec(fb_param->split_point(), &(param_.get()->split_point));
    ppl::nn::pmx::utils::Fbvec2Stdvec(fb_param->constant_split_data(), &constant_split_data_);
    
    return CommonInit();
}
#endif

}}}}} // namespace ppl::nn::llm::cuda::pmx
