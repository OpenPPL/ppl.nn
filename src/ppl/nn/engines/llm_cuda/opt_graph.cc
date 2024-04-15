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

#include "kernel.h"
#include "opt_graph.h"
#include "opt_kernel_creator_manager.h"
#include "opt_pass_manager.h"

#include "ppl/nn/engines/utils.h" // LoadConstants()
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace llm { namespace cuda {

static RetCode CreateOps(const ir::GraphTopo* topo, map<nodeid_t, unique_ptr<OptKernel>>* ops) {
    for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        auto& type = node->GetType();
        auto creator = OptKernelCreatorManager::GetInstance()->Find(type.domain, type.name, type.version);
        if (!creator) {
            LOG(ERROR) << "cannot find creator for [" << node->GetName() << "] of type [" << type.domain
                       << ":" << type.name << ":" << type.version << "].";
            return RC_NOT_FOUND;
        }

        auto op = unique_ptr<LlmCudaOptKernel>((*creator)(node));
        if (!op) {
            LOG(ERROR) << "create op for [" << node->GetName() << "] failed.";
            return RC_OUT_OF_MEMORY;
        }

        ops->emplace(node->GetId(), std::move(op));
    }

    return RC_SUCCESS;
}

RetCode OptGraph::Init(const utils::SharedResource& resource, ir::Graph* graph, RuntimePartitionInfo* partition_info) {
    graph_ = graph;
    partition_info_ = partition_info;

    RetCode rc;

    rc = CreateOps(graph_->topo.get(), &partition_info_->kernels);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "create ops failed.";
        return rc;
    }

    return rc;
}

static RetCode InitOps(std::map<nodeid_t, std::unique_ptr<OptKernel>>& kernels, OptKernelOptions& options) {
    for (auto it = kernels.begin(); it != kernels.end(); ++it) {
        auto kernel = (LlmCudaOptKernel*)(it->second.get());
        auto status = kernel->Init(options);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "init op for [" << kernel->GetNode()->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

RetCode OptGraph::Optimize(
    const utils::SharedResource& resource,
    const EngineOptions& engine_options,
    LlmCudaDevice* device)
{
    OptKernelOptions options;
    options.resource = &resource;
    options.graph = graph_;
    options.device = device;
    options.partition_info = partition_info_;
    options.engine_options = &engine_options;

    RetCode rc;

    rc = InitOps(partition_info_->kernels, options);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "InitOps failed: " << GetRetCodeStr(rc);
        return rc;
    }

    if (engine_options.quant_method == QUANT_METHOD_ONLINE_I8I8) {
        LOG(INFO) << "Processing I8I8Quantization...";
        auto prc = OptPassManager::GetInstance()->Apply("", "I8I8Quantization", options);
        if (prc.retcode != RC_SUCCESS) {
            LOG(ERROR) << "I8I8Quantization failed: " << GetRetCodeStr(prc.retcode);
            return prc.retcode;
        }
        if (!prc.graph_modified) {
            LOG(INFO) << "I8I8Quantization: nothing has been changed.";
        }
        rc = OptPassManager::GetInstance()->ApplyByDomain("i8i8.fuse", options);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "I8I8Fuse failed: " << GetRetCodeStr(rc);
            return rc;
        }
    }

    if (engine_options.quant_method == QUANT_METHOD_ONLINE_I4F16) {
        LOG(INFO) << "I4F16Quantization has not been implemented";
        return ppl::common::RC_UNSUPPORTED;
    }

    rc = utils::LoadConstants(*graph_, device, &partition_info_->constants);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "LoadConstants failed: " << GetRetCodeStr(rc);
        return rc;
    }

    return rc;
}

}}}}
