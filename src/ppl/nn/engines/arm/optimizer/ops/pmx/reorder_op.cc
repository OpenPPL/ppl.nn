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

#include "ppl/nn/engines/arm/optimizer/ops/pmx/reorder_op.h"
#include "ppl/nn/engines/arm/kernels/pmx/reorder_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

ReorderOp::ReorderOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        auto& input = *info->GetInput<TensorImpl>(0)->GetShape();
        auto& output = *info->GetOutput<TensorImpl>(0)->GetShape();
        if ((output.GetDataFormat() == DATAFORMAT_N4CX || output.GetDataFormat() == DATAFORMAT_N8CX) 
            && input.GetDimCount() == 2) {
            int64_t new_dims[4] = {input.GetDim(0), input.GetDim(1), 1, 1};
            output.Reshape(new_dims, 4);
        } else if (output.GetDataFormat() == DATAFORMAT_N8CX && input.GetDimCount() < 3) {
            auto padded_output_shape = PadShapeTo3Dims(input);
            output.Reshape(padded_output_shape.GetDims(), padded_output_shape.GetDimCount());
        } else {
            if (input.IsScalar()) {
                output.ReshapeAsScalar();
            } else {
                output.Reshape(input.GetDims(), input.GetDimCount());
            }
        }
        return RC_SUCCESS;
    };

    infer_type_func_ = PassiveInferType;
}

RetCode ReorderOp::Init(const OptKernelOptions& options) {
    return RC_SUCCESS;
}

#ifdef PPLNN_ENABLE_PMX_MODEL

ppl::common::RetCode ReorderOp::SerializeData(const ::ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder output_builder;
    auto fb_output_data = ppl::nn::pmx::arm::CreateOutputDataDirect(output_builder, &common_param_.output_types, &common_param_.output_formats);
    auto fb_op_data = ppl::nn::pmx::arm::CreateOpData(output_builder, ppl::nn::pmx::arm::PrivateDataType_OutputData, fb_output_data.Union());
    ppl::nn::pmx::arm::FinishOpDataBuffer(output_builder, fb_op_data);

    flatbuffers::FlatBufferBuilder op_builder;
    auto fb_data = op_builder.CreateVector(output_builder.GetBufferPointer(), output_builder.GetSize());
    auto fb_root = ppl::nn::pmx::onnx::CreateOpParam(op_builder, ppl::nn::pmx::onnx::OpParamType_NONE, 0, fb_data);
    ppl::nn::pmx::onnx::FinishOpParamBuffer(op_builder, fb_root);
    return ds->Write(op_builder.GetBufferPointer(), op_builder.GetSize());
}

ppl::common::RetCode ReorderOp::DeserializeData(const ::ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    auto fb_op_param = ppl::nn::pmx::onnx::GetOpParam(base);
    auto arm_op_data = ppl::nn::pmx::arm::GetOpData(fb_op_param->data_()->data());
    
    auto arm_output_data = arm_op_data->value_as_OutputData();
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_output_data->dtype(), &common_param_.output_types);
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_output_data->dformat(), &common_param_.output_formats);

    return RC_SUCCESS;
}

#endif

KernelImpl* ReorderOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ReorderKernel>();
}

}}} // namespace ppl::nn::arm
