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

#include "pixel_unshuffle_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/opmx/pixel_unshuffle_kernel.h"
#include "ppl/nn/oputils/opmx/reshape_pixel_unshuffle.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/llm_cuda/pmx/generated/llm_cuda_op_params_generated.h"
#endif

using namespace std;
using namespace ppl::common;


namespace ppl { namespace nn { namespace llm { namespace cuda { namespace opmx {

RetCode PixelUnshuffleOp::CommonInit() {
    infer_type_and_format_func_ = GenericInferTypeAndFormat;
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return nn::opmx::ReshapePixelUnshuffle(info, param_.get());
    };
    return RC_SUCCESS;
}

RetCode PixelUnshuffleOp::DoInit(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ppl::nn::opmx::PixelUnshuffleParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenericLoadParam failed: " << GetRetCodeStr(status);
        return status;
    }

    return CommonInit();
}

KernelImpl* PixelUnshuffleOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<PixelUnshuffleKernel>(param_.get());
}

#ifdef PPLNN_ENABLE_PMX_MODEL
ppl::common::RetCode PixelUnshuffleOp::SerializeData(const ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder builder;
    auto fb_param = opmx::CreatePixelUnshuffleParam(builder,
        param_.get()->scale_factor,
        param_.get()->data_layout);
    auto fb_op_param = opmx::CreateOpParam(builder, opmx::OpParamType_PixelUnshuffleParam, fb_param.Union());
    opmx::FinishOpParamBuffer(builder, fb_op_param);
    return ds->Write(builder.GetBufferPointer(), builder.GetSize());
}

ppl::common::RetCode PixelUnshuffleOp::DeserializeData(const ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    auto fb_op_param = opmx::GetOpParam(base);
    auto fb_param = fb_op_param->value_as_PixelUnshuffleParam();
    param_ = make_shared<ppl::nn::opmx::PixelUnshuffleParam>();
    param_.get()->scale_factor              = fb_param->scale_factor();
    param_.get()->data_layout               = fb_param->data_layout();

    return CommonInit();
}
#endif


}}}}} // namespace ppl::nn::llm::cuda::opmx
