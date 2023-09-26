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

#include "engine.h"
#include "engine_context.h"
#include "plain_device.h"
#include "buffered_device.h"
#include "opt_graph.h"
#include "opt_kernel_creator_manager.h"

#include "ppl/nn/engines/llm_cuda/engine_factory.h"
#include "ppl/nn/common/logger.h"

#include <stdarg.h>
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace llm { namespace cuda {

RetCode LlmCudaEngine::Init(const EngineOptions& options) {
    options_ = options;
    device_.reset(new PlainDevice());
    auto rc = device_->Init(options.device_id, false, false, &tensor_parallel_nccl_param_);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "init device failed: " << GetRetCodeStr(rc);
    }
    return rc;
}

EngineContext* LlmCudaEngine::CreateEngineContext() {
    auto ctx = unique_ptr<LlmCudaEngineContext>(new LlmCudaEngineContext());
    if (ctx) {
        auto rc = ctx->Init(options_, &tensor_parallel_nccl_param_);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "init engine context failed: " << GetRetCodeStr(rc);
            return nullptr;
        }
    }

    return ctx.release();
}

RetCode LlmCudaEngine::ProcessGraph(const utils::SharedResource& resource, ir::Graph* graph, RuntimePartitionInfo* info) {
    OptGraph opt_graph;

    auto status = opt_graph.Init(resource, graph, info);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "OptGraph Init failed: " << GetRetCodeStr(status);
        return status;
    }

    status = opt_graph.Optimize(resource, options_, device_.get());
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "OptGraph Optimize failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

EngineImpl* LlmCudaEngine::Create() {
    return static_cast<EngineImpl*>(EngineFactory::Create(options_));
}

bool LlmCudaEngine::Supports(const ir::Node* node) const {
    auto& type = node->GetType();
    return (OptKernelCreatorManager::GetInstance()->Find(type.domain, type.name, type.version) != nullptr);
}

/* ------------------------------------------------------------------------- */

RetCode LlmCudaEngine::SetTensorParellelNcclComm(LlmCudaEngine* engine, va_list args) {
#ifdef PPLNN_CUDA_ENABLE_NCCL
    auto nccl_comm = va_arg(args, ncclComm_t);
    engine->tensor_parallel_nccl_param_.comm = nccl_comm;
    NCCL_CHECK(ncclCommCount(nccl_comm, &engine->tensor_parallel_nccl_param_.size), "ncclCommCount");
    NCCL_CHECK(ncclCommUserRank(nccl_comm, &engine->tensor_parallel_nccl_param_.rank), "ncclCommUserRank");
    LOG(INFO) << "TP NCCL world size: " << engine->tensor_parallel_nccl_param_.size;
    return RC_SUCCESS;
#else
    LOG(ERROR) << "Please recompile with NCCL support.";
    return RC_INVALID_VALUE;
#endif
}

LlmCudaEngine::ConfHandlerFunc LlmCudaEngine::conf_handlers_[] = {
    SetTensorParellelNcclComm,
};

RetCode LlmCudaEngine::Configure(uint32_t option, ...) {
    if (option >= ENGINE_CONF_MAX) {
        LOG(ERROR) << "invalid option[" << option << "] >= [" << (uint32_t)ENGINE_CONF_MAX << "]";
        return RC_INVALID_VALUE;
    }
    va_list args;
    va_start(args, option);
    auto status = conf_handlers_[option](this, args);
    va_end(args);

    return status;
}

}}}} // namespace ppl::nn::llm::cuda
