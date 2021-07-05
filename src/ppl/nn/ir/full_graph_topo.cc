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

#include "ppl/nn/ir/full_graph_topo.h"
#include "ppl/nn/utils/vector_utils.h"
#include <algorithm>
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace ir {

static Node* FindNode(const string& name, const unique_ptr<Node>* nodes, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        auto node = nodes[i].get();
        if (node && node->GetName() == name) {
            return node;
        }
    }
    return nullptr;
}

static Edge* FindEdge(const string& name, const unique_ptr<Edge>* edges, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        auto edge = edges[i].get();
        if (edge && edge->GetName() == name) {
            return edge;
        }
    }
    return nullptr;
}

pair<Node*, bool> FullGraphTopo::AddNode(const string& name) {
    auto node = FindNode(name, nodes_.data(), nodes_.size());
    if (node) {
        return make_pair(node, false);
    }

    node = new Node(nodes_.size());
    node->SetName(name);
    nodes_.emplace_back(unique_ptr<Node>(node));
    return make_pair(node, true);
}

Node* FullGraphTopo::GetNodeById(nodeid_t nid) {
    if (nid >= nodes_.size()) {
        return nullptr;
    }
    return nodes_[nid].get();
}

const Node* FullGraphTopo::GetNodeById(nodeid_t nid) const {
    if (nid >= nodes_.size()) {
        return nullptr;
    }
    return nodes_[nid].get();
}

void FullGraphTopo::DelNodeById(nodeid_t nid) {
    if (nid < nodes_.size() && nodes_[nid]) {
        nodes_[nid].reset();
    }
}

class FullGraphEdge final : public Edge {
public:
    FullGraphEdge(edgeid_t id) : id_(id), producer_(INVALID_NODEID) {}

    edgeid_t GetId() const override {
        return id_;
    }

    void SetName(const std::string& name) override {
        name_ = name;
    }
    const std::string& GetName() const override {
        return name_;
    }

    nodeid_t GetProducer() const override {
        return producer_;
    }
    void SetProducer(nodeid_t p) override {
        producer_ = p;
    }

    ConsumerIter CreateConsumerIter() const override {
        return ConsumerIter(&consumers_);
    }
    uint32_t CalcConsumerCount() const override {
        return consumers_.size();
    }

    void AddConsumer(nodeid_t nid) override {
        auto it = std::find(consumers_.begin(), consumers_.end(), nid);
        if (it == consumers_.end()) {
            consumers_.push_back(nid);
        }
    }

    bool DelConsumer(nodeid_t nid) override {
        auto old_size = consumers_.size();
        utils::VectorRemoveOneIf(consumers_, [nid](nodeid_t id) -> bool {
            return (id == nid);
        });
        return (consumers_.size() != old_size);
    }

    void ClearConsumer() override {
        consumers_.clear();
    }

private:
    const edgeid_t id_;
    std::string name_;
    nodeid_t producer_;
    std::vector<nodeid_t> consumers_;

private:
    FullGraphEdge(const FullGraphEdge&) = delete;
    void operator=(const FullGraphEdge&) = delete;
};

pair<Edge*, bool> FullGraphTopo::AddEdge(const string& name) {
    auto edge = FindEdge(name, edges_.data(), edges_.size());
    if (edge) {
        return make_pair(edge, false);
    }

    edge = new FullGraphEdge(GetMaxEdgeId());
    edge->SetName(name);
    edges_.emplace_back(unique_ptr<Edge>(edge));
    return make_pair(edge, true);
}

Edge* FullGraphTopo::GetEdgeById(edgeid_t eid) {
    if (eid >= edges_.size()) {
        return nullptr;
    }
    return edges_[eid].get();
}

const Edge* FullGraphTopo::GetEdgeById(edgeid_t eid) const {
    if (eid >= edges_.size()) {
        return nullptr;
    }
    return edges_[eid].get();
}

void FullGraphTopo::DelEdgeById(edgeid_t eid) {
    if (eid >= edges_.size()) {
        return;
    }

    auto p = [eid](edgeid_t id) -> bool {
        return (eid == id);
    };

    utils::VectorRemoveAllIf(inputs_, p);
    utils::VectorRemoveAllIf(extra_inputs_, p);
    utils::VectorRemoveAllIf(outputs_, p);
    utils::VectorRemoveAllIf(constants_, p);

    edges_[eid].reset();
}

}}} // namespace ppl::nn::ir
