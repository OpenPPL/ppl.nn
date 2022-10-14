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

#include "ppl/nn/engines/arm/optimizer/ops/pmx/channel_shuffle_op.h"
#include "ppl/nn/engines/arm/kernels/pmx/channel_shuffle_kernel.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/arm/pmx/generated/arm_op_params_generated.h"
#endif

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

ChannelShuffleOp::ChannelShuffleOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_type_func_ = GenericInferType;

    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        auto& input0 = *info->GetInput<TensorImpl>(0)->GetShape();
        int64_t channels = input0.GetDim(1);
        for (uint32_t i = 1; i < info->GetInputCount(); ++i) {
            channels += info->GetInput<TensorImpl>(1)->GetShape()->GetDim(1);
        }
        if (channels % info->GetOutputCount()) {
            return ppl::common::RC_INVALID_VALUE;
        }
        channels /= info->GetOutputCount();
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto& output = *info->GetOutput<TensorImpl>(i)->GetShape();
            output.Reshape(input0.GetDims(), input0.GetRealDimCount());
            output.SetDim(1, channels);
        }

        return RC_SUCCESS;
    };
}

RetCode ChannelShuffleOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode ChannelShuffleOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                                       vector<dataformat_t>* selected_output_formats) {
    auto data_format = info.GetInput<TensorImpl>(0)->GetShape()->GetDataFormat();
    if (info.GetInputCount() == 2 && info.GetInput<TensorImpl>(1)->GetShape()->GetDataFormat() != data_format) {
        data_format = DATAFORMAT_NDARRAY;
    }
    for (uint32_t i = 0; i < info.GetInputCount(); i++) {
        selected_input_formats->at(i) = data_format;
    }
    for (uint32_t i = 0; i < info.GetOutputCount(); i++) {
        selected_output_formats->at(i) = data_format;
    }

    return RC_SUCCESS;
}

#ifdef PPLNN_ENABLE_PMX_MODEL

ppl::common::RetCode ChannelShuffleOp::SerializeData(const ::ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder chsh_param_builder;
    auto fb_chsh_op_param = ppl::nn::pmx::arm::CreateChannelShuffleParam(chsh_param_builder, param_->group);
    auto fb_pmx_op_data = ppl::nn::pmx::arm::CreatePmxOpDataDirect(chsh_param_builder, &common_param_.output_types, &common_param_.output_formats, ppl::nn::pmx::arm::PmxOpType_ChannelShuffleParam, fb_chsh_op_param.Union());
    auto fb_op_data = ppl::nn::pmx::arm::CreateOpData(chsh_param_builder, ppl::nn::pmx::arm::PrivateDataType_PmxOpData, fb_pmx_op_data.Union());
    ppl::nn::pmx::arm::FinishOpDataBuffer(chsh_param_builder, fb_op_data);

    flatbuffers::FlatBufferBuilder op_builder;
    auto fb_data = op_builder.CreateVector(chsh_param_builder.GetBufferPointer(), chsh_param_builder.GetSize());
    auto fb_root = ppl::nn::pmx::onnx::CreateOpParam(op_builder, ppl::nn::pmx::onnx::OpParamType_NONE, 0, fb_data);
    ppl::nn::pmx::onnx::FinishOpParamBuffer(op_builder, fb_root);
    return ds->Write(op_builder.GetBufferPointer(), op_builder.GetSize());
    return 0;
}

ppl::common::RetCode ChannelShuffleOp::DeserializeData(const ::ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {

    auto fb_op_param = ppl::nn::pmx::onnx::GetOpParam(base);
    param_ = std::make_shared<ppl::nn::pmx::ChannelShuffleParam>();

    auto arm_op_data = ppl::nn::pmx::arm::GetOpData(fb_op_param->data_()->data());
    
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_op_data->value_as_PmxOpData()->dtype(), &common_param_.output_types);
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_op_data->value_as_PmxOpData()->dformat(), &common_param_.output_formats);
    auto arm_shape_op_param = arm_op_data->value_as_PmxOpData()->value_as_ChannelShuffleParam();
    param_->group = arm_shape_op_param->group();
    return RC_SUCCESS;
}

#endif

KernelImpl* ChannelShuffleOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ChannelShuffleKernel>(param_.get());
}

}}} // namespace ppl::nn::arm
