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

#include "ppl/nn/ir/graph_topo.h"
#include "ppl/nn/ir/utils.h"
#include "ppl/nn/utils/vector_utils.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace ir {

static Node* FindNode(const ir::GraphTopo* topo, const string& name) {
    for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetName() == name) {
            return node;
        }
    }
    return nullptr;
}

Node* GraphTopo::GetNodeByName(const string& name) {
    return FindNode(this, name);
}

const Node* GraphTopo::GetNodeByName(const string& name) const {
    return FindNode(this, name);
}

static edgeid_t FindEdgeId(const string& name, const vector<edgeid_t>& edge_ids, const GraphTopo* topo) {
    for (uint32_t i = 0; i < edge_ids.size(); ++i) {
        auto eid = edge_ids[i];
        auto edge = topo->GetEdgeById(eid);
        if (edge && edge->GetName() == name) {
            return eid;
        }
    }
    return INVALID_EDGEID;
}

Edge* GraphTopo::GetEdgeByName(const std::string& name) {
    for (auto it = CreateEdgeIter(); it->IsValid(); it->Forward()) {
        auto edge = it->Get();
        if (edge->GetName() == name) {
            return edge;
        }
    }
    return nullptr;
}

const Edge* GraphTopo::GetEdgeByName(const std::string& name) const {
    for (auto it = CreateEdgeIter(); it->IsValid(); it->Forward()) {
        auto edge = it->Get();
        if (edge->GetName() == name) {
            return edge;
        }
    }
    return nullptr;
}

edgeid_t GraphTopo::GetInput(const string& name) const {
    return FindEdgeId(name, inputs_, this);
}

edgeid_t GraphTopo::GetConstant(const string& name) const {
    return FindEdgeId(name, constants_, this);
}

edgeid_t GraphTopo::GetOutput(const string& name) const {
    return FindEdgeId(name, outputs_, this);
}

edgeid_t GraphTopo::GetExtraInput(const string& name) const {
    return FindEdgeId(name, extra_inputs_, this);
}

void GraphTopo::MarkAsInput(edgeid_t eid) {
    utils::VectorAddUnique(inputs_, eid);
}

void GraphTopo::MarkAsOutput(edgeid_t eid) {
    utils::VectorAddUnique(outputs_, eid);
}

void GraphTopo::MarkAsExtraInput(edgeid_t eid) {
    utils::VectorAddUnique(extra_inputs_, eid);
}

void GraphTopo::MarkAsConstant(edgeid_t eid) {
    utils::VectorAddUnique(constants_, eid);
}

RetCode GraphTopo::ReplaceWithNode(const string& node_name, const Node::Type& node_type) {
    auto ret_pair = AddNode(node_name);
    if (!ret_pair.second) {
        return RC_EXISTS;
    }
    auto node = ret_pair.first;

    node->SetType(node_type);

    for (auto it = inputs_.begin(); it != inputs_.end(); ++it) {
        GetEdgeById(*it)->ClearConsumer();
    }
    for (auto it = extra_inputs_.begin(); it != extra_inputs_.end(); ++it) {
        GetEdgeById(*it)->ClearConsumer();
    }
    for (auto it = constants_.begin(); it != constants_.end(); ++it) {
        GetEdgeById(*it)->ClearConsumer();
    }
    // some outputs may be constants or intermediate edges
    for (auto it = outputs_.begin(); it != outputs_.end(); ++it) {
        GetEdgeById(*it)->ClearConsumer();
    }

    vector<bool> reserved_edges(GetMaxEdgeId(), false);

    for (auto it = inputs_.begin(); it != inputs_.end(); ++it) {
        auto edge = GetEdgeById(*it);
        node->AddInput(*it);
        edge->AddConsumer(node->GetId());
        reserved_edges[*it] = true;
    }
    for (auto it = extra_inputs_.begin(); it != extra_inputs_.end(); ++it) {
        auto edge = GetEdgeById(*it);
        node->AddInput(*it); // all extra inputs are treated as normal inputs
        edge->AddConsumer(node->GetId());
        reserved_edges[*it] = true;
    }
    for (auto it = constants_.begin(); it != constants_.end(); ++it) {
        auto edge = GetEdgeById(*it);
        node->AddInput(*it); // all constants are treated as normal inputs
        edge->AddConsumer(node->GetId());
        reserved_edges[*it] = true;
    }
    for (auto it = outputs_.begin(); it != outputs_.end(); ++it) {
        auto edge = GetEdgeById(*it);
        node->AddOutput(*it);
        edge->SetProducer(node->GetId());
        reserved_edges[*it] = true;
    }

    for (edgeid_t i = 0; i < reserved_edges.size(); ++i) {
        if (!reserved_edges[i]) {
            DelEdgeById(i);
        }
    }

    for (nodeid_t i = 0; i < node->GetId(); ++i) {
        DelNodeById(i);
    }

    return RC_SUCCESS;
}

static void DoFindPredecessors(edgeid_t eid, const GraphTopo* topo, vector<nodeid_t>* res) {
    auto edge = topo->GetEdgeById(eid);
    auto pid = edge->GetProducer();
    if (pid != INVALID_NODEID) {
        if (std::find(res->begin(), res->end(), pid) == res->end()) {
            res->push_back(pid);
        }
    }
}

vector<nodeid_t> GraphTopo::FindPredecessors(nodeid_t nid) const {
    auto node = GetNodeById(nid);

    vector<nodeid_t> res;
    for (uint32_t i = 0; i < node->GetInputCount(); ++i) {
        auto eid = node->GetInput(i); // INVALID_EDGEID means nil input
        if (eid != INVALID_EDGEID) {
            DoFindPredecessors(eid, this, &res);
        }
    }
    for (uint32_t i = 0; i < node->GetExtraInputCount(); ++i) {
        auto eid = node->GetExtraInput(i); // INVALID_EDGEID means nil input
        if (eid != INVALID_EDGEID) {
            DoFindPredecessors(eid, this, &res);
        }
    }
    return res;
}

vector<nodeid_t> GraphTopo::FindSuccessors(nodeid_t nid) const {
    auto node = GetNodeById(nid);

    vector<nodeid_t> res;
    for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
        auto eid = node->GetOutput(i);
        auto edge = GetEdgeById(eid);
        for (auto it = edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
            auto nid = it.Get();
            if (std::find(res.begin(), res.end(), nid) == res.end()) {
                res.push_back(nid);
            }
        }
    }
    return res;
}

void GraphTopo::TopologicalSort(const function<void(nodeid_t)>& callback) const {
    auto node_iter = CreateNodeIter();
    utils::Dfs(
        GetMaxNodeId(),
        [&node_iter]() -> nodeid_t {
            if (node_iter->IsValid()) {
                auto ret = node_iter->Get();
                node_iter->Forward();
                return ret->GetId();
            }
            return INVALID_NODEID;
        },
        [this](nodeid_t nid, const function<void(nodeid_t)>& f) -> void {
            auto prevs = this->FindPredecessors(nid);
            for (auto x = prevs.begin(); x != prevs.end(); ++x) {
                f(*x);
            }
        },
        callback);
}

}}} // namespace ppl::nn::ir
