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

#include "ppl/nn/ir/partial_graph_topo.h"
#include "ppl/nn/utils/vector_utils.h"
#include <set>
using namespace std;

namespace ppl { namespace nn { namespace ir {

class PartialGraphEdge final : public Edge {
public:
    PartialGraphEdge(Edge* parent_edge, const vector<Node*>* node_ptrs)
        : parent_edge_(parent_edge), node_ptrs_(node_ptrs) {}

    edgeid_t GetId() const override {
        return parent_edge_->GetId();
    }

    void SetName(const string& name) override {
        parent_edge_->SetName(name);
    }

    const string& GetName() const override {
        return parent_edge_->GetName();
    }

    void SetProducer(nodeid_t p) override {
        parent_edge_->SetProducer(p);
    }

    nodeid_t GetProducer() const override {
        auto pid = parent_edge_->GetProducer();
        if (pid < node_ptrs_->size() && node_ptrs_->at(pid)) {
            return pid;
        }
        return INVALID_NODEID;
    }

    Edge::ConsumerIter CreateConsumerIter() const override {
        auto iter = parent_edge_->CreateConsumerIter();
        iter.Reset([this](nodeid_t nid) -> bool {
            return (nid < node_ptrs_->size() && node_ptrs_->at(nid));
        });
        return iter;
    }

    uint32_t CalcConsumerCount() const override {
        uint32_t count = 0;
        for (auto it = CreateConsumerIter(); it.IsValid(); it.Forward()) {
            ++count;
        }
        return count;
    }

    void AddConsumer(nodeid_t nid) override {
        parent_edge_->AddConsumer(nid);
    }

    void DelConsumer(nodeid_t nid) override {
        if (nid < node_ptrs_->size() && node_ptrs_->at(nid)) {
            parent_edge_->DelConsumer(nid);
        }
    }

    void ClearConsumer() override {
        vector<nodeid_t> node_ids;
        for (auto it = CreateConsumerIter(); it.IsValid(); it.Forward()) {
            node_ids.push_back(it.Get());
        }

        for (auto it = node_ids.begin(); it != node_ids.end(); ++it) {
            parent_edge_->DelConsumer(*it);
        }
    }

private:
    Edge* parent_edge_;
    const vector<Node*>* node_ptrs_;
};

PartialGraphTopo::PartialGraphTopo(GraphTopo* parent, const vector<nodeid_t>& nodes) {
    parent_ = parent;

    node_ptrs_.resize(parent->GetCurrentNodeIdBound(), nullptr);
    edge_ptrs_.resize(parent->GetCurrentEdgeIdBound(), nullptr);

    // set valid node and edge ptrs from parent
    for (uint32_t i = 0; i < nodes.size(); ++i) {
        auto node = parent->GetNode(nodes[i]);
        node_ptrs_[nodes[i]] = node;

        for (uint32_t j = 0; j < node->GetInputCount(); ++j) {
            auto eid = node->GetInput(j);
            if (eid != INVALID_EDGEID) {
                edge_ptrs_[eid] = parent->GetEdge(eid);
            }
        }
        for (uint32_t j = 0; j < node->GetOutputCount(); ++j) {
            auto eid = node->GetOutput(j);
            edge_ptrs_[eid] = parent->GetEdge(eid);
        }
    }

    set<edgeid_t> constants;
    for (uint32_t i = 0; i < parent->GetConstantCount(); ++i) {
        constants.insert(parent->GetConstant(i));
    }

    // override input/output edges
    for (uint32_t eid = 0; eid < edge_ptrs_.size(); ++eid) {
        auto edge = edge_ptrs_[eid];
        if (!edge) {
            continue;
        }

        if (constants.find(edge->GetId()) != constants.end()) {
            constants_.push_back(edge->GetId());
            continue;
        }

        auto producer_id = edge->GetProducer();
        // has outer producer, marked as input
        if (producer_id >= node_ptrs_.size() || !node_ptrs_[producer_id]) {
            auto new_edge = new PartialGraphEdge(edge, &node_ptrs_);
            override_edges_.emplace(eid, unique_ptr<Edge>(new_edge));
            edge_ptrs_[eid] = new_edge;
            inputs_.push_back(eid);
        } else {
            for (auto it = edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
                auto consumer_id = it.Get();
                // has outer consumer, marked as output
                if (consumer_id >= node_ptrs_.size() || !node_ptrs_[consumer_id]) {
                    auto new_edge = new PartialGraphEdge(edge, &node_ptrs_);
                    override_edges_.emplace(eid, unique_ptr<Edge>(new_edge));
                    edge_ptrs_[eid] = new_edge;
                    outputs_.push_back(eid);
                    break;
                }
            }
        }
    }

    // handling nodes' extra inputs
    for (uint32_t i = 0; i < nodes.size(); ++i) {
        auto node = parent->GetNode(nodes[i]);
        for (uint32_t j = 0; j < node->GetExtraInputCount(); ++j) {
            auto eid = node->GetExtraInput(j);
            // if an extra input of a node is not a normal edge, it is an extra input of the graph
            if (eid != INVALID_EDGEID && !edge_ptrs_[eid]) {
                auto edge = parent->GetEdge(eid);
                auto new_edge = new PartialGraphEdge(edge, &node_ptrs_);
                override_edges_.emplace(eid, unique_ptr<Edge>(new_edge));
                edge_ptrs_[eid] = new_edge;
                extra_inputs_.push_back(eid);
            }
        }
    }
}

pair<Node*, bool> PartialGraphTopo::AddNode(const string& name) {
    auto ret_pair = parent_->AddNode(name);
    auto node = ret_pair.first;

    node_ptrs_.resize(parent_->GetCurrentNodeIdBound(), nullptr);

    if (ret_pair.second) {
        node_ptrs_[node->GetId()] = node;
    } else {
        if (node) {
            ret_pair.first = node_ptrs_[node->GetId()];
        } else {
            ret_pair.first = nullptr;
        }
    }

    return ret_pair;
}

Node* PartialGraphTopo::GetNode(nodeid_t nid) const {
    if (nid >= node_ptrs_.size()) {
        return nullptr;
    }
    return node_ptrs_[nid];
}

void PartialGraphTopo::DelNode(nodeid_t nid) {
    if (node_ptrs_[nid]) {
        node_ptrs_[nid] = nullptr;
        parent_->DelNode(nid);
    }
}

pair<Edge*, bool> PartialGraphTopo::AddEdge(const string& name) {
    auto ret_pair = parent_->AddEdge(name);
    auto edge = ret_pair.first;

    edge_ptrs_.resize(parent_->GetCurrentEdgeIdBound(), nullptr);

    if (ret_pair.second) {
        edge_ptrs_[edge->GetId()] = edge;
    } else {
        if (edge) {
            ret_pair.first = edge_ptrs_[edge->GetId()];
        } else {
            ret_pair.first = nullptr;
        }
    }

    return ret_pair;
}

Edge* PartialGraphTopo::GetEdge(edgeid_t eid) const {
    if (eid >= edge_ptrs_.size()) {
        return nullptr;
    }
    return edge_ptrs_[eid];
}

void PartialGraphTopo::DelEdge(edgeid_t eid) {
    if (eid < edge_ptrs_.size()) {
        if (edge_ptrs_[eid]) {
            edge_ptrs_[eid] = nullptr;
            parent_->DelEdge(eid);
            override_edges_.erase(eid);
        }
    }
}

}}} // namespace ppl::nn::ir
