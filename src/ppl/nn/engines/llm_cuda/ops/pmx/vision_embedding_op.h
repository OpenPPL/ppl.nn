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

#ifndef _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_OPS_PMX_VISION_EMBEDDING_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_OPS_PMX_VISION_EMBEDDING_OP_H_

#include "ppl/nn/engines/llm_cuda/opt_kernel.h"
#include "ppl/nn/params/pmx/vision_embedding_param.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

class VisionEmbeddingOp final : public LlmCudaOptKernel {
public:
    VisionEmbeddingOp(const ir::Node* node) : LlmCudaOptKernel(node) {}

    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode DoInit(const OptKernelOptions&) override;

#ifdef PPLNN_ENABLE_PMX_MODEL
    ppl::common::RetCode SerializeData(const ppl::nn::pmx::SerializationContext&, utils::DataStream*) const override;
    ppl::common::RetCode DeserializeData(const ppl::nn::pmx::DeserializationContext&, const void*, uint64_t) override;
#endif

private:
    ppl::common::RetCode CommonInit();

    std::shared_ptr<ppl::nn::pmx::VisionEmbeddingParam> param_;
};

}}}}} // namespace ppl::nn::llm::cuda::pmx

#endif
