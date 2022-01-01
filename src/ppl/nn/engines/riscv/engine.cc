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

#include "ppl/nn/engines/riscv/riscv_device.h"
#include "ppl/nn/engines/riscv/engine.h"
#include "ppl/nn/engines/riscv/kernel.h"
#include "ppl/nn/engines/riscv/engine_context.h"
#include "ppl/nn/engines/riscv/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/riscv/optimizer/opt_graph.h"
#include "ppl/nn/runtime/runtime_partition_info.h"
#include "ppl/nn/engines/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

ppl::common::RetCode RiscvEngine::Init(const RiscvEngineOptions& options) {
    options_ = options;
    return ppl::common::RC_SUCCESS;
};

EngineContext* RiscvEngine::CreateEngineContext() {
    return new RiscvEngineContext(&device_);
}

ppl::common::RetCode RiscvEngine::Configure(uint32_t, ...) {
    return ppl::common::RC_UNSUPPORTED;
}

bool RiscvEngine::Supports(const ir::Node* node) const {
    auto& type = node->GetType();
    bool ok = OptKernelCreatorManager::Instance()->Find(type.domain, type.name, type.version) != nullptr;
    return ok;
}

RetCode RiscvEngine::DoOptimize(ir::Graph* graph, utils::SharedResource* resource, RuntimePartitionInfo* info) {
    OptGraph opt_graph;
    auto status = opt_graph.Init(graph, resource, info, &options_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init OptGraph failed: " << GetRetCodeStr(status);
        return status;
    }

    status = opt_graph.DoOptimize(&device_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "OptGraph DoOptimize failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

ppl::common::RetCode RiscvEngine::CalDataOmittedConstants(const ir::Graph& graph, const RuntimePartitionInfo& info,
                                                          std::set<edgeid_t>* data_omitted_constants) const {
    data_omitted_constants->clear();

    std::map<edgeid_t, int64_t> constants_data_refcount;
    for (uint32_t i = 0; i < graph.topo->GetConstantCount(); ++i) {
        auto edge_id = graph.topo->GetConstant(i);
        auto edge = graph.topo->GetEdgeById(edge_id);
        if (edge == nullptr) {
            LOG(ERROR) << "Edge of Constant[edgeid=" << edge_id << "] not found";
            return RC_NOT_FOUND;
        }
        const int64_t refcount = edge->CalcConsumerCount();
        auto ret = constants_data_refcount.insert(make_pair(edge_id, refcount));
        if (!ret.second) {
            LOG(ERROR) << "Duplicated Constant Edge";
            return ppl::common::RC_OTHER_ERROR;
        }
    }

    for (auto it = info.kernels.begin(); it != info.kernels.end(); ++it) {
        auto kernel = (RiscvOptKernel*)it->second.get();
        auto ret = kernel->OmitConstantsData(&constants_data_refcount);
        if (ppl::common::RC_SUCCESS != ret) {
            return ret;
        }
    }

    for (auto it = constants_data_refcount.begin(); it != constants_data_refcount.end(); ++it) {
        if (it->second <= 0) {
            data_omitted_constants->insert(it->first);
        }
    }

    return ppl::common::RC_SUCCESS;
}

RetCode RiscvEngine::ProcessGraph(utils::SharedResource* resource, ir::Graph* graph, RuntimePartitionInfo* info) {
    auto status = DoOptimize(graph, resource, info);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "DoOptimize failed: " << GetRetCodeStr(status);
        return status;
    }

    std::set<edgeid_t> data_omitted_constants;
    status = CalDataOmittedConstants(*graph, *info, &data_omitted_constants);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "CalDataOmittedConstants failed: " << GetRetCodeStr(status);
        return status;
    }

    status = utils::LoadConstants(*graph, &device_, &info->constants);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "LoadConstants failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::riscv
