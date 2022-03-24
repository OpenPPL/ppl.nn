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

#include "ppl/nn/runtime/partial_runtime_creator.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/ir/partial_graph_topo.h"
#include "ppl/nn/ir/utils.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

static void InitOpName2NodeId(const ir::GraphTopo* topo, const std::vector<nodeid_t>& nids,
                              map<string, uint32_t>* name2nid) {
    for (uint32_t i = 0; i < nids.size(); ++i) {
        auto node = topo->GetNodeById(nids[i]);
        name2nid->insert(make_pair(string(node->GetName()), nids[i]));
    }
}

void PartialRuntimeCreator::Init(const ir::GraphTopo* topo, const std::shared_ptr<RuntimeGraphInfo>& info,
                                 const std::shared_ptr<RuntimeAuxInfo>& aux_info) {
    topo_ = topo;
    graph_info_ = info;
    aux_info_ = aux_info;
    InitOpName2NodeId(topo, aux_info_->sorted_nodes, &name2nid_);
}

template <typename ContainerType>
static RetCode ConvertName2NodeId(const char** names, uint32_t name_num, const map<string, uint32_t>& name2nid,
                                  ContainerType* nids) {
    for (uint32_t i = 0; i < name_num; ++i) {
        auto ref = name2nid.find(names[i]);
        if (ref == name2nid.end()) {
            LOG(ERROR) << "cannot find op[" << names[i] << "]";
            return RC_NOT_FOUND;
        }
        nids->push_back(ref->second);
    }
    return RC_SUCCESS;
}

void PartialRuntimeCreator::InitBeginEndOps(const char** begin_ops, uint32_t begin_op_num, const char** end_ops,
                                            uint32_t end_op_num, const map<string, uint32_t>& name2nid,
                                            BeginEndOps* ops) {
    ConvertName2NodeId(begin_ops, begin_op_num, name2nid, &ops->begin);
    ConvertName2NodeId(end_ops, end_op_num, name2nid, &ops->end);
}

RetCode PartialRuntimeCreator::InitPartialRuntimeResource(const ir::GraphTopo* topo,
                                                          const set<nodeid_t>& reserved_edgeids, const BeginEndOps& ops,
                                                          PartialRuntimeResource* resource) {
    vector<nodeid_t> sorted_nodes;
    utils::ReversedDfs(
        topo->GetMaxNodeId(),
        [&ops](const function<void(nodeid_t)>& f) -> void {
            for (auto x = ops.end.begin(); x != ops.end.end(); ++x) {
                f(*x);
            }
        },
        [topo](nodeid_t nid, const function<void(nodeid_t)>& f) -> void {
            auto prevs = topo->FindPredecessors(nid);
            for (auto x = prevs.begin(); x != prevs.end(); ++x) {
                f(*x);
            }
        },
        [&sorted_nodes](nodeid_t nid) -> void {
            sorted_nodes.push_back(nid);
        },
        [&ops](nodeid_t current) -> bool {
            return (std::find(ops.begin.begin(), ops.begin.end(), current) != ops.begin.end());
        });

    resource->topo = make_shared<ir::PartialGraphTopo>(const_cast<ir::GraphTopo*>(topo), sorted_nodes);

    for (auto it = resource->topo->CreateEdgeIter(); it->IsValid(); it->Forward()) {
        auto edge = it->Get();
        if (reserved_edgeids.find(edge->GetId()) != reserved_edgeids.end()) {
            resource->reserved_edgeids.insert(edge->GetId());
        }
    }

    resource->aux_info = make_shared<RuntimeAuxInfo>();
    auto status = GenerateRuntimeAuxInfo(resource->topo.get(), resource->reserved_edgeids, resource->aux_info.get());
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init RuntimeAuxInfo failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RuntimeImpl* PartialRuntimeCreator::Create(const char** begin_ops, uint32_t begin_op_num, const char** end_ops,
                                           uint32_t end_op_num, const set<nodeid_t>& reserved_edgeids) {
    BeginEndOps ops;
    InitBeginEndOps(begin_ops, begin_op_num, end_ops, end_op_num, name2nid_, &ops);

    auto ret_pair = ops2resource_.insert(make_pair(ops, PartialRuntimeResource()));
    PartialRuntimeResource* resource = &ret_pair.first->second;
    if (ret_pair.second) {
        auto status = InitPartialRuntimeResource(topo_, reserved_edgeids, ops, resource);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "InitPartialRuntimeResource failed: " << GetRetCodeStr(status);
            return nullptr;
        }
    }

    auto runtime = new RuntimeImpl();
    auto status = runtime->Init(resource->topo, graph_info_, resource->aux_info, resource->reserved_edgeids);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init runtime failed: " << GetRetCodeStr(status);
        delete runtime;
        return nullptr;
    }

    return runtime;
}

}} // namespace ppl::nn
