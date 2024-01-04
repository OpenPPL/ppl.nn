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

#ifndef _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_ENGINE_H_
#define _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_ENGINE_H_

#include "llm_cuda_device.h"

#include "ppl/nn/engines/llm_cuda/engine_options.h"
#include "ppl/nn/engines/engine_impl.h"
#include "ppl/nn/utils/shared_resource.h"

#include <memory>

namespace ppl { namespace nn { namespace llm { namespace cuda {

class LlmCudaEngine final : public EngineImpl {
public:
    LlmCudaEngine() : EngineImpl("llm_cuda") {}
    ppl::common::RetCode Init(const EngineOptions&);
    ppl::common::RetCode Configure(uint32_t, ...) override;
    EngineContext* CreateEngineContext() override;
    bool Supports(const ir::Node*) const override;
    ppl::common::RetCode ProcessGraph(const utils::SharedResource&, ir::Graph*, RuntimePartitionInfo*) override;
    EngineImpl* Create() override;
    ppl::common::NcclParam* GetTensorParallelNcclParam() { return &tensor_parallel_nccl_param_; };

#ifdef PPLNN_ENABLE_PMX_MODEL
    ppl::common::RetCode LoadConstants(const ConstantVisitor&, std::map<edgeid_t, BufferInfo>*) override;
    OptKernel* CreateOptKernel(const ir::Node*) const override;
    ppl::common::RetCode SerializeData(const ppl::nn::pmx::SerializationContext&, utils::DataStream*) const override;
    ppl::common::RetCode DeserializeData(const void*, uint64_t) override;
#endif

private:
    static ppl::common::RetCode SetTensorParellelNcclComm(LlmCudaEngine*, va_list);

    typedef ppl::common::RetCode (*ConfHandlerFunc)(LlmCudaEngine*, va_list);
    static ConfHandlerFunc conf_handlers_[ENGINE_CONF_MAX];

private:
    EngineOptions options_;
    ppl::common::NcclParam tensor_parallel_nccl_param_;
    std::unique_ptr<LlmCudaDevice> device_;
};

}}}} // namespace ppl::nn::llm::cuda

#endif
