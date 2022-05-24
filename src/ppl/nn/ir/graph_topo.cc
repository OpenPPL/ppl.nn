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
#include <queue>
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

Node* GraphTopo::GetNode(const string& name) const {
    return FindNode(this, name);
}

static edgeid_t FindEdgeId(const string& name, const vector<edgeid_t>& edge_ids, const GraphTopo* topo) {
    for (uint32_t i = 0; i < edge_ids.size(); ++i) {
        auto eid = edge_ids[i];
        auto edge = topo->GetEdge(eid);
        if (edge && edge->GetName() == name) {
            return eid;
        }
    }
    return INVALID_EDGEID;
}

Edge* GraphTopo::GetEdge(const std::string& name) const {
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
        GetEdge(*it)->ClearConsumer();
    }
    for (auto it = extra_inputs_.begin(); it != extra_inputs_.end(); ++it) {
        GetEdge(*it)->ClearConsumer();
    }
    for (auto it = constants_.begin(); it != constants_.end(); ++it) {
        GetEdge(*it)->ClearConsumer();
    }
    // some outputs may be constants or intermediate edges
    for (auto it = outputs_.begin(); it != outputs_.end(); ++it) {
        GetEdge(*it)->ClearConsumer();
    }

    vector<bool> reserved_edges(GetCurrentEdgeIdBound(), false);

    // normal inputs
    for (auto it = inputs_.begin(); it != inputs_.end(); ++it) {
        auto eid = *it;
        if (!reserved_edges[eid]) {
            auto edge = GetEdge(eid);
            node->AddInput(eid);
            edge->AddConsumer(node->GetId());
            reserved_edges[eid] = true;
        }
    }
    // extra inputs are treated as normal inputs.
    for (auto it = extra_inputs_.begin(); it != extra_inputs_.end(); ++it) {
        auto eid = *it;
        if (!reserved_edges[eid]) {
            auto edge = GetEdge(eid);
            node->AddInput(eid);
            edge->AddConsumer(node->GetId());
            reserved_edges[eid] = true;
        }
    }
    // constants are treated as normal inputs
    for (auto it = constants_.begin(); it != constants_.end(); ++it) {
        auto eid = *it;
        if (!reserved_edges[eid]) {
            auto edge = GetEdge(eid);
            node->AddInput(eid);
            edge->AddConsumer(node->GetId());
            reserved_edges[eid] = true;
        }
    }

    for (auto it = outputs_.begin(); it != outputs_.end(); ++it) {
        auto eid = *it;
        auto edge = GetEdge(eid);
        node->AddOutput(eid);
        edge->SetProducer(node->GetId());
        reserved_edges[eid] = true;
    }

    // remove unused intermediate edges
    for (edgeid_t i = 0; i < reserved_edges.size(); ++i) {
        if (!reserved_edges[i]) {
            DelEdge(i);
        }
    }

    // id of newly inserted node is the max value of all nodes
    for (nodeid_t i = 0; i < node->GetId(); ++i) {
        DelNode(i);
    }

    return RC_SUCCESS;
}

static void DoFindPredecessors(edgeid_t eid, const GraphTopo* topo, vector<nodeid_t>* res) {
    auto edge = topo->GetEdge(eid);
    auto pid = edge->GetProducer();
    if (pid != INVALID_NODEID) {
        if (std::find(res->begin(), res->end(), pid) == res->end()) {
            res->push_back(pid);
        }
    }
}

vector<nodeid_t> GraphTopo::FindPredecessors(nodeid_t nid) const {
    auto node = GetNode(nid);

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
    auto node = GetNode(nid);

    vector<nodeid_t> res;
    for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
        auto eid = node->GetOutput(i);
        auto edge = GetEdge(eid);
        for (auto it = edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
            auto nid = it.Get();
            if (std::find(res.begin(), res.end(), nid) == res.end()) {
                res.push_back(nid);
            }
        }
    }
    return res;
}

set<nodeid_t> GraphTopo::FindAncestors(nodeid_t nid) const {
    set<nodeid_t> dedup;
    queue<nodeid_t> q;

    q.push(nid);
    while (!q.empty()) {
        nid = q.front();
        q.pop();
        auto node = GetNode(nid);

        for (uint32_t i = 0; i < node->GetInputCount(); ++i) {
            auto eid = node->GetInput(i);
            if (eid != INVALID_EDGEID) {
                auto edge = GetEdge(eid);
                auto pid = edge->GetProducer();
                if (pid != INVALID_NODEID) {
                    auto ret_pair = dedup.insert(pid);
                    if (ret_pair.second) {
                        q.push(pid);
                    }
                }
            }
        }

        for (uint32_t i = 0; i < node->GetExtraInputCount(); ++i) {
            auto eid = node->GetExtraInput(i);
            if (eid != INVALID_EDGEID) {
                auto edge = GetEdge(eid);
                auto pid = edge->GetProducer();
                if (pid != INVALID_NODEID) {
                    auto ret_pair = dedup.insert(pid);
                    if (ret_pair.second) {
                        q.push(pid);
                    }
                }
            }
        }
    }

    return dedup;
}

vector<nodeid_t> GraphTopo::FindLeafNodes() const {
    vector<nodeid_t> leaf_nodes;
    for (auto it = CreateEdgeIter(); it->IsValid(); it->Forward()) {
        auto edge = it->Get();
        auto producer_id = edge->GetProducer();
        if (producer_id != INVALID_NODEID && edge->CalcConsumerCount() == 0) {
            leaf_nodes.push_back(producer_id);
        }
    }
    return leaf_nodes;
}

void GraphTopo::TopologicalSort(const function<void(nodeid_t)>& callback) const {
    utils::ReversedDfs(
        GetCurrentNodeIdBound(),
        [this](const function<void(nodeid_t)>& f) -> void {
            auto leaf_nodes = FindLeafNodes();
            for (auto x = leaf_nodes.begin(); x != leaf_nodes.end(); ++x) {
                f(*x);
            }
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
