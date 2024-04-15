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
#include "ppl/nn/engines/utils.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/llm_cuda/pmx/generated/llm_cuda_engine_generated.h"
#endif

#include <stdarg.h>
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace llm { namespace cuda {

RetCode LlmCudaEngine::Init(const EngineOptions& options) {
    options_ = options;
    device_.reset(new PlainDevice(false));
    auto rc = device_->Init(options.device_id, false, &tensor_parallel_nccl_param_, DeviceStreamFlag::NONE);
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

#ifdef PPLNN_ENABLE_PMX_MODEL
RetCode LlmCudaEngine::LoadConstants(const ConstantVisitor& visitor, map<edgeid_t, BufferInfo>* eid2info) {
    return utils::LoadConstants(visitor, device_.get(), eid2info);
}

OptKernel* LlmCudaEngine::CreateOptKernel(const ir::Node* node) const {
    auto& type = node->GetType();
    auto creator = OptKernelCreatorManager::GetInstance()->Find(type.domain, type.name, type.version);
    if (!creator) {
        LOG(ERROR) << "cannot find creator for node[" << node->GetName() << "] of type[" << type.domain << ":"
                   << type.name << ":" << type.version << "]";
        return nullptr;
    }

    auto opt_kernel = (*creator)(node);
    if (!opt_kernel) {
        LOG(ERROR) << "create kernel[" << node->GetName() << "] failed: oom.";
        return nullptr;
    }

    return opt_kernel;
}

ppl::common::RetCode LlmCudaEngine::SerializeData(const pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder builder;
    auto fb_param = CreateEngineOptionsParam(builder, options_.cublas_layout_hint, GetVersion());
    auto fb_engine_param = CreateEngineParam(builder, EngineParamType_EngineOptionsParam, fb_param.Union());
    FinishEngineParamBuffer(builder, fb_engine_param);
    return ds->Write(builder.GetBufferPointer(), builder.GetSize());
}

ppl::common::RetCode LlmCudaEngine::DeserializeData(const void* base, uint64_t size) {
    auto fb_engine_param = GetEngineParam(base);
    auto fb_param = fb_engine_param->value_as_EngineOptionsParam();
    
    uint32_t cublas_layout_hint = fb_param->cublas_layout_hint();
    if (cublas_layout_hint != options_.cublas_layout_hint) {
        LOG(WARNING) << "deserialize cublas_layout_hint[" << cublas_layout_hint << "] diff from user input[" <<  options_.cublas_layout_hint << "]";
    }
    options_.cublas_layout_hint = cublas_layout_hint;
    
    if (fb_param->version() != GetVersion()) {
        LOG(WARNING) << "engine version[" << GetVersion() << "] diff from pmx version[" <<  fb_param->version() << "]";
    }

    return ppl::common::RC_SUCCESS;
}
#endif

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
