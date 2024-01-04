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

#include "parallel_embedding_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/pmx/parallel_embedding_kernel.h"
#include "ppl/nn/oputils/pmx/reshape_parallel_embedding.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/engines/llm_cuda/engine.h"
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/llm_cuda/pmx/generated/llm_cuda_op_params_generated.h"
#endif

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;


namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

RetCode ParallelEmbeddingOp::CommonInit() {
    infer_type_and_format_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto weight_shape = info->GetInput<TensorImpl>(1)->GetShape();
        auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();

        output_shape->SetDataFormat(DATAFORMAT_NDARRAY);
        output_shape->SetDataType(weight_shape->GetDataType());

        return RC_SUCCESS;
    };
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return nn::pmx::ReshapeParallelEmbedding(info, param_.get(), nccl_param_->size);
    };
    return RC_SUCCESS;
}

RetCode ParallelEmbeddingOp::DoInit(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ppl::nn::pmx::ParallelEmbeddingParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenericLoadParam failed: " << GetRetCodeStr(status);
        return status;
    }
    nccl_param_ = options.device->GetTensorParallelNcclParam();

    return CommonInit();
}

KernelImpl* ParallelEmbeddingOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ParallelEmbeddingKernel>(param_.get());
}

#ifdef PPLNN_ENABLE_PMX_MODEL
ppl::common::RetCode ParallelEmbeddingOp::SerializeData(const ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder builder;
    auto fb_param = pmx::CreateParallelEmbeddingParam(builder, 
        param_.get()->num_embeddings,
        param_.get()->embedding_dims,
        param_.get()->padding_idx,
        param_.get()->max_norm,
        param_.get()->norm_type);
    auto fb_op_param = pmx::CreateOpParam(builder, pmx::OpParamType_ParallelEmbeddingParam, fb_param.Union());
    pmx::FinishOpParamBuffer(builder, fb_op_param);
    return ds->Write(builder.GetBufferPointer(), builder.GetSize());
}

ppl::common::RetCode ParallelEmbeddingOp::DeserializeData(const ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    auto fb_op_param = pmx::GetOpParam(base);
    auto fb_param = fb_op_param->value_as_ParallelEmbeddingParam();
    param_ = make_shared<ppl::nn::pmx::ParallelEmbeddingParam>();
    param_.get()->num_embeddings = fb_param->num_embeddings();
    param_.get()->embedding_dims = fb_param->embedding_dims();
    param_.get()->padding_idx    = fb_param->padding_idx();
    param_.get()->max_norm       = fb_param->max_norm();
    param_.get()->norm_type      = fb_param->norm_type();
    
    nccl_param_ = dynamic_cast<LlmCudaEngine*>(ctx.engine)->GetTensorParallelNcclParam();

    return CommonInit();
}
#endif


}}}}} // namespace ppl::nn::llm::cuda::pmx
