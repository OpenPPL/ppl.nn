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

#include "vision_embedding_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/opmx/vision_embedding_kernel.h"
#include "ppl/nn/oputils/opmx/reshape_vision_embedding.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/engines/llm_cuda/engine.h"
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/llm_cuda/pmx/generated/llm_cuda_op_params_generated.h"
#endif

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::opmx;


namespace ppl { namespace nn { namespace llm { namespace cuda { namespace opmx {

RetCode VisionEmbeddingOp::CommonInit() {
    infer_type_and_format_func_ = GenericInferTypeAndFormat;
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return nn::opmx::ReshapeVisionEmbedding(info, param_.get());
    };
    return RC_SUCCESS;
}

RetCode VisionEmbeddingOp::DoInit(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ppl::nn::opmx::VisionEmbeddingParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenericLoadParam failed: " << GetRetCodeStr(status);
        return status;
    }

    return CommonInit();
}

KernelImpl* VisionEmbeddingOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<VisionEmbeddingKernel>(param_.get());
}

#ifdef PPLNN_ENABLE_PMX_MODEL
ppl::common::RetCode VisionEmbeddingOp::SerializeData(const ppl::nn::pmx::SerializationContext&, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder builder;
    auto fb_param = opmx::CreateVisionEmbeddingParam(builder,
        param_.get()->hidden_dim,
        param_.get()->image_size,
        param_.get()->patch_size);
    auto fb_op_param = opmx::CreateOpParam(builder, opmx::OpParamType_VisionEmbeddingParam, fb_param.Union());
    opmx::FinishOpParamBuffer(builder, fb_op_param);
    return ds->Write(builder.GetBufferPointer(), builder.GetSize());
}

ppl::common::RetCode VisionEmbeddingOp::DeserializeData(const ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    auto fb_op_param = opmx::GetOpParam(base);
    auto fb_param = fb_op_param->value_as_VisionEmbeddingParam();
    param_ = make_shared<ppl::nn::opmx::VisionEmbeddingParam>();
    param_.get()->hidden_dim = fb_param->hidden_dim();
    param_.get()->image_size = fb_param->image_size();
    param_.get()->patch_size = fb_param->patch_size();

    return CommonInit();
}
#endif


}}}}} // namespace ppl::nn::llm::cuda::opmx
