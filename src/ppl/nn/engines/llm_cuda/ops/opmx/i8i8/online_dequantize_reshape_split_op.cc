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

#include "online_dequantize_reshape_split_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/opmx/i8i8/online_dequantize_reshape_split_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_split.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/engines/llm_cuda/engine.h"
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/llm_cuda/pmx/generated/llm_cuda_op_i8i8_params_generated.h"
#endif

using namespace std;
using namespace ppl::common;


namespace ppl { namespace nn { namespace llm { namespace cuda { namespace opmx {

RetCode I8I8OnlineDequantizeReshapeSplitOp::CommonInit() {
    infer_type_and_format_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto scale_outer_shape = info->GetInput<TensorImpl>(2)->GetShape();

        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto output_shape = info->GetOutput<TensorImpl>(i)->GetShape();
            output_shape->SetDataFormat(DATAFORMAT_NDARRAY);
            output_shape->SetDataType(scale_outer_shape->GetDataType());
        }

        return RC_SUCCESS;
    };
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        std::vector<int64_t> output_dims(param_.shape.begin(), param_.shape.end());
        auto input_shape = info->GetInput<TensorImpl>(0)->GetShape();

        const int32_t axis = param_.split_param->axis < 0
            ? param_.split_param->axis + param_.shape.size()
            : param_.split_param->axis;

        // fill zeros
        for (int32_t i = 0; i < axis; ++i) {
            output_dims[i] = input_shape->GetDim(i);
        }
        // set split dim
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            output_dims[axis] = param_.split[i];
            auto output_shape = info->GetOutput<TensorImpl>(i)->GetShape();
            output_shape->Reshape(output_dims);
        }

        return ppl::common::RC_SUCCESS;
    };

    return ppl::common::RC_SUCCESS;
}

RetCode I8I8OnlineDequantizeReshapeSplitOp::DoInit(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ppl::nn::onnx::SplitParam>(options, &param_.split_param);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenericLoadParam failed: " << GetRetCodeStr(status);
        return status;
    }

    return CommonInit();
}

KernelImpl* I8I8OnlineDequantizeReshapeSplitOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<I8I8OnlineDequantizeReshapeSplitKernel>(&param_);
}

#ifdef PPLNN_ENABLE_PMX_MODEL
ppl::common::RetCode I8I8OnlineDequantizeReshapeSplitOp::SerializeData(const ppl::nn::pmx::SerializationContext&, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder builder;
    auto fb_split_point = builder.CreateVector(param_.split_param.get()->split_point);
    auto fb_split = builder.CreateVector(param_.split);
    auto fb_shape = builder.CreateVector(param_.shape);
    auto fb_param = opmx::i8i8::CreateOnlineDequantizeReshapeSplitParam(builder, 
        param_.split_param.get()->axis,
        fb_split_point,
        fb_split,
        fb_shape,
        param_.bias_term);
    auto fb_op_param = opmx::i8i8::CreateOpParam(builder, opmx::i8i8::OpParamType_OnlineDequantizeReshapeSplitParam, fb_param.Union());
    opmx::i8i8::FinishOpParamBuffer(builder, fb_op_param);
    return ds->Write(builder.GetBufferPointer(), builder.GetSize());
}

ppl::common::RetCode I8I8OnlineDequantizeReshapeSplitOp::DeserializeData(const ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    auto fb_op_param = opmx::i8i8::GetOpParam(base);
    auto fb_param = fb_op_param->value_as_OnlineDequantizeReshapeSplitParam();
    param_.split_param = make_shared<ppl::nn::onnx::SplitParam>();
    param_.split_param.get()->axis = fb_param->axis();
    param_.bias_term = fb_param->bias_term();
    ppl::nn::opmx::utils::Fbvec2Stdvec(fb_param->split_point(), &(param_.split_param.get()->split_point));
    ppl::nn::opmx::utils::Fbvec2Stdvec(fb_param->split(), &(param_.split));
    ppl::nn::opmx::utils::Fbvec2Stdvec(fb_param->shape(), &(param_.shape));

    return CommonInit();
}
#endif


}}}}} // namespace ppl::nn::llm::cuda::opmx
