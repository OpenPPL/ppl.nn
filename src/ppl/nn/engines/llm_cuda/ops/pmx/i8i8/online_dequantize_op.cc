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

#include "online_dequantize_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/pmx/i8i8/online_dequantize_kernel.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/engines/llm_cuda/engine.h"
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/llm_cuda/pmx/generated/llm_cuda_op_i8i8_params_generated.h"
#endif

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::llm::cuda;

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

RetCode I8I8OnlineDequantizeOp::CommonInit() {
    infer_type_and_format_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto scale_outer_shape = info->GetInput<TensorImpl>(1)->GetShape();
        auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();

        output_shape->SetDataFormat(DATAFORMAT_NDARRAY);
        output_shape->SetDataType(scale_outer_shape->GetDataType());

        return RC_SUCCESS;
    };
    infer_dims_func_ = GenericInferDims;
    return RC_SUCCESS;
}

RetCode I8I8OnlineDequantizeOp::DoInit(const OptKernelOptions& options) {
    return CommonInit();
}

KernelImpl* I8I8OnlineDequantizeOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<I8I8OnlineDequantizeKernel>(&param_);
}

#ifdef PPLNN_ENABLE_PMX_MODEL
ppl::common::RetCode I8I8OnlineDequantizeOp::SerializeData(const ppl::nn::pmx::SerializationContext&, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder builder;
    auto fb_param = pmx::i8i8::CreateOnlineDequantizeParam(builder, param_.bias_term);
    auto fb_op_param = pmx::i8i8::CreateOpParam(builder, pmx::i8i8::OpParamType_OnlineDequantizeParam, fb_param.Union());
    pmx::i8i8::FinishOpParamBuffer(builder, fb_op_param);
    return ds->Write(builder.GetBufferPointer(), builder.GetSize());
}

ppl::common::RetCode I8I8OnlineDequantizeOp::DeserializeData(const ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    auto fb_op_param = pmx::i8i8::GetOpParam(base);
    auto fb_param = fb_op_param->value_as_OnlineDequantizeParam();
    param_.bias_term = fb_param->bias_term();
    
    return CommonInit();
}
#endif

}}}}} // namespace ppl::nn::llm::cuda::pmx
