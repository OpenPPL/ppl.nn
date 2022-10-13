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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/resize_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/resize_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_resize.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/oputils/onnx/resize.h"
#endif

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

ResizeOp::ResizeOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeResize(info, param_.get());
    };

    infer_type_func_ = GenericInferType;
}

RetCode ResizeOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode ResizeOp::SelectDataType(const InputOutputInfo& info,
                                 std::vector<ppl::common::datatype_t>* selected_input_types,
                                 std::vector<ppl::common::datatype_t>* selected_output_types,
                                 const ppl::common::datatype_t preferred_fp_datatype) {
    GenericSelectDataType(info, selected_input_types, selected_output_types, preferred_fp_datatype);
    const int64_t input_count = info.GetInputCount();
    if (input_count >= 2) {
        selected_input_types->at(1) = ppl::common::DATATYPE_FLOAT32;
    }
    if (input_count >= 3) {
        selected_input_types->at(2) = ppl::common::DATATYPE_FLOAT32;
    }
    if (input_count >= 4) {
        selected_input_types->at(3) = ppl::common::DATATYPE_INT64;
    }
    return RC_SUCCESS;
}

RetCode ResizeOp::SelectFormat(const InputOutputInfo& info,
                               std::vector<ppl::common::dataformat_t>* selected_input_formats,
                               std::vector<ppl::common::dataformat_t>* selected_output_formats) {
    const auto input_shape = info.GetInput<TensorImpl>(0)->GetShape();
    auto selected_format = input_shape->GetDataFormat();
    if (param_->mode == param_->RESIZE_MODE_CUBIC) {
        const auto input_type = input_shape->GetDataType();
        const int64_t channels = input_shape->GetDimCount() == 4 ? input_shape->GetDim(1) : 0;
        if (input_type == ppl::common::DATATYPE_FLOAT32 && channels >= 2) {
            selected_format = ppl::common::DATAFORMAT_N4CX;
        } else if (input_type == ppl::common::DATATYPE_FLOAT16 && channels >= 4) {
            selected_format = ppl::common::DATAFORMAT_N8CX;
        }
    }
    selected_input_formats->at(0) = selected_output_formats->at(0) = selected_format;
    return RC_SUCCESS;
}

#ifdef PPLNN_ENABLE_PMX_MODEL

ppl::common::RetCode ResizeOp::SerializeData(const ::ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder output_builder;
    auto fb_output_data = ppl::nn::pmx::arm::CreateOutputDataDirect(output_builder, &common_param_.output_types, &common_param_.output_formats);
    auto fb_op_data = ppl::nn::pmx::arm::CreateOpData(output_builder, ppl::nn::pmx::arm::PrivateDataType_OutputData, fb_output_data.Union());
    ppl::nn::pmx::arm::FinishOpDataBuffer(output_builder, fb_op_data);

    flatbuffers::FlatBufferBuilder op_builder;
    auto fb_param = ppl::nn::pmx::onnx::SerializeResizeParam(*param_.get(), &op_builder);
    auto fb_data = op_builder.CreateVector(output_builder.GetBufferPointer(), output_builder.GetSize());
    auto fb_root = ppl::nn::pmx::onnx::CreateOpParam(op_builder, ppl::nn::pmx::onnx::OpParamType_ResizeParam, fb_param.Union(), fb_data);
    ppl::nn::pmx::onnx::FinishOpParamBuffer(op_builder, fb_root);
    return ds->Write(op_builder.GetBufferPointer(), op_builder.GetSize());
}

ppl::common::RetCode ResizeOp::DeserializeData(const ::ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    auto fb_op_param = ppl::nn::pmx::onnx::GetOpParam(base);

    param_ = std::make_shared<ppl::nn::onnx::ResizeParam>();
    ppl::nn::pmx::onnx::DeserializeResizeParam(*fb_op_param->value_as_ResizeParam(), param_.get());

    auto arm_op_data = ppl::nn::pmx::arm::GetOpData(fb_op_param->data_()->data());
    auto arm_output_data = arm_op_data->value_as_OutputData();
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_output_data->dtype(), &common_param_.output_types);
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_output_data->dformat(), &common_param_.output_formats);
    return RC_SUCCESS;
}

#endif

KernelImpl* ResizeOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ResizeKernel>(param_.get());
}

}}} // namespace ppl::nn::arm
