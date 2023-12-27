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

#ifndef _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_ENGINE_CONTEXT_H_
#define _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_ENGINE_CONTEXT_H_

#include "llm_cuda_device.h"

#include "ppl/nn/engines/llm_cuda/engine_options.h"
#include "ppl/nn/engines/engine_context.h"
#include "ppl/common/cuda/nccl_utils.h"

namespace ppl { namespace nn { namespace llm { namespace cuda {

class LlmCudaEngineContext final : public EngineContext {
public:
    LlmCudaEngineContext() {}
    ppl::common::RetCode Init(const EngineOptions& options, ppl::common::NcclParam* tensor_parallel_nccl_param);

    Device* GetDevice() const override {
        return device_.get();
    }
    const char* GetName() const override {
        return "llm_cuda";
    }
    const EngineOptions& GetEngineOptions() const {
        return engine_options_;
    }

private:
    std::shared_ptr<LlmCudaDevice> device_;
    EngineOptions engine_options_;

private:
    LlmCudaEngineContext(const LlmCudaEngineContext&) = delete;
    LlmCudaEngineContext& operator=(const LlmCudaEngineContext&) = delete;
};

}}}} // namespace ppl::nn::llm::cuda

#endif
