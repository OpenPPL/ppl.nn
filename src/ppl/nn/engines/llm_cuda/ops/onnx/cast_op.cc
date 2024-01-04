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

#include "cast_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/onnx/cast_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_cast.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/cast.h"
#endif

using namespace std;
using namespace ppl::common;


namespace ppl { namespace nn { namespace llm { namespace cuda { namespace onnx {

RetCode CastOp::CommonInit() {
    infer_type_and_format_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto input_shape = info->GetInput<TensorImpl>(0)->GetShape();
        auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();
        output_shape->SetDataType(param_->to);
        output_shape->SetDataFormat(input_shape->GetDataFormat());
        return ppl::common::RC_SUCCESS;
    };
    infer_dims_func_ = GenericInferDims;
    return RC_SUCCESS;
}

RetCode CastOp::DoInit(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ppl::nn::onnx::CastParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenericLoadParam failed: " << GetRetCodeStr(status);
        return status;
    }

    return CommonInit();
}

KernelImpl* CastOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<CastKernel>(param_.get());
}

#ifdef PPLNN_ENABLE_PMX_MODEL
ppl::common::RetCode CastOp::SerializeData(const pmx::SerializationContext&, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder builder;
    auto fb_param = pmx::onnx::SerializeCastParam(*param_, &builder);
    auto fb_op_param = pmx::onnx::CreateOpParam(builder, pmx::onnx::OpParamType_CastParam, fb_param.Union());
    pmx::onnx::FinishOpParamBuffer(builder, fb_op_param);
    return ds->Write(builder.GetBufferPointer(), builder.GetSize());
}

ppl::common::RetCode CastOp::DeserializeData(const pmx::DeserializationContext&, const void* base, uint64_t size) {
    auto fb_op_param = pmx::onnx::GetOpParam(base);
    auto fb_argmax_param = fb_op_param->value_as_CastParam();
    param_ = make_shared<ppl::nn::onnx::CastParam>();
    pmx::onnx::DeserializeCastParam(*fb_argmax_param, param_.get());
    
    return CommonInit();
}
#endif

}}}}} // namespace ppl::nn::llm::cuda::pmx
