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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/concat_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/concat_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_concat.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/oputils/onnx/concat.h"
#endif

#include <set>

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

ConcatOp::ConcatOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeConcat(info, param_.get());
    };

    infer_type_func_ = GenericInferType;
}

RetCode ConcatOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode ConcatOp::SelectFormat(const InputOutputInfo& info,
                               std::vector<ppl::common::dataformat_t>* selected_input_formats,
                               std::vector<ppl::common::dataformat_t>* selected_output_formats) {
    const uint32_t input_count = info.GetInputCount();
    std::set<dataformat_t> input_dataformats;
    for (uint32_t i = 0; i < input_count; i++) {
        input_dataformats.insert(info.GetInput<TensorImpl>(i)->GetShape()->GetDataFormat());
    }

    dataformat_t data_format;
    if (input_dataformats.size() == 1) { // all input has same dataformat
        data_format = *input_dataformats.begin();
    } else { // all data format fall back to NDARRAY
        data_format = DATAFORMAT_NDARRAY;
    }

    for (uint32_t i = 0; i < input_count; i++) {
        selected_input_formats->at(i) = data_format;
    }
    selected_output_formats->at(0) = data_format;

    return RC_SUCCESS;
}

#ifdef PPLNN_ENABLE_PMX_MODEL

ppl::common::RetCode ConcatOp::SerializeData(const ::ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder output_builder;
    auto fb_output_data = ppl::nn::pmx::arm::CreateOutputDataDirect(output_builder, &common_param_.output_types, &common_param_.output_formats);
    auto fb_op_data = ppl::nn::pmx::arm::CreateOpData(output_builder, ppl::nn::pmx::arm::PrivateDataType_OutputData, fb_output_data.Union());
    ppl::nn::pmx::arm::FinishOpDataBuffer(output_builder, fb_op_data);

    flatbuffers::FlatBufferBuilder op_builder;
    auto fb_param = ppl::nn::pmx::onnx::SerializeConcatParam(*param_.get(), &op_builder);
    auto fb_data = op_builder.CreateVector(output_builder.GetBufferPointer(), output_builder.GetSize());
    auto fb_root = ppl::nn::pmx::onnx::CreateOpParam(op_builder, ppl::nn::pmx::onnx::OpParamType_ConcatParam, fb_param.Union(), fb_data);
    ppl::nn::pmx::onnx::FinishOpParamBuffer(op_builder, fb_root);
    return ds->Write(op_builder.GetBufferPointer(), op_builder.GetSize());
}

ppl::common::RetCode ConcatOp::DeserializeData(const ::ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    auto fb_op_param = ppl::nn::pmx::onnx::GetOpParam(base);

    param_ = std::make_shared<ppl::nn::onnx::ConcatParam>();
    ppl::nn::pmx::onnx::DeserializeConcatParam(*fb_op_param->value_as_ConcatParam(), param_.get());

    auto arm_op_data = ppl::nn::pmx::arm::GetOpData(fb_op_param->data_()->data());
    auto arm_output_data = arm_op_data->value_as_OutputData();
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_output_data->dtype(), &common_param_.output_types);
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_output_data->dformat(), &common_param_.output_formats);
    return RC_SUCCESS;
}

#endif

KernelImpl* ConcatOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ConcatKernel>(param_.get());
}

}}} // namespace ppl::nn::arm
