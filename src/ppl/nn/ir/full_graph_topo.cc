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

pair<Node*, bool> FullGraphTopo::AddNode(const string& name) {
    auto ret_pair = name2nid_.insert(make_pair(name, GetCurrentNodeIdBound()));
    if (!ret_pair.second) {
        return make_pair(nodes_[ret_pair.first->second].get(), false);
    }

    auto node = new Node(ret_pair.first->second, &ret_pair.first->first);
    nodes_.emplace_back(unique_ptr<Node>(node));
    return make_pair(node, true);
}

Node* FullGraphTopo::GetNode(nodeid_t nid) const {
    if (nid >= nodes_.size()) {
        return nullptr;
    }
    return nodes_[nid].get();
}

void FullGraphTopo::DelNode(nodeid_t nid) {
    if (nid < nodes_.size() && nodes_[nid]) {
        name2nid_.erase(nodes_[nid]->GetName());
        nodes_[nid].reset();
    }
}

class FullGraphEdge final : public Edge {
public:
    FullGraphEdge(edgeid_t id, const string* shared_name_str)
        : id_(id), shared_name_str_(shared_name_str), producer_(INVALID_NODEID) {}

    edgeid_t GetId() const override {
        return id_;
    }

    const string& GetName() const override {
        return *shared_name_str_;
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

    void DelConsumer(nodeid_t nid) override {
        utils::VectorRemoveOneIf(consumers_, [nid](nodeid_t id) -> bool {
            return (id == nid);
        });
    }

    void ClearConsumer() override {
        consumers_.clear();
    }

private:
    friend class FullGraphTopo; // for GraphTopo::RenameEdge()

    const edgeid_t id_;
    const string* shared_name_str_; // pointer to GraphTopo::name2eid_[name]::first
    nodeid_t producer_;
    std::vector<nodeid_t> consumers_;

private:
    FullGraphEdge(const FullGraphEdge&) = delete;
    void operator=(const FullGraphEdge&) = delete;
};

pair<Edge*, bool> FullGraphTopo::AddEdge(const string& name) {
    auto ret_pair = name2eid_.insert(make_pair(name, GetCurrentEdgeIdBound()));
    if (!ret_pair.second) {
        return make_pair(edges_[ret_pair.first->second].get(), false);
    }

    auto edge = new FullGraphEdge(ret_pair.first->second, &ret_pair.first->first);
    edges_.emplace_back(unique_ptr<Edge>(edge));
    return make_pair(edge, true);
}

Edge* FullGraphTopo::GetEdge(edgeid_t eid) const {
    if (eid >= edges_.size()) {
        return nullptr;
    }
    return edges_[eid].get();
}

void FullGraphTopo::DelEdge(edgeid_t eid) {
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

    name2eid_.erase(edges_[eid]->GetName());
    edges_[eid].reset();
}

bool FullGraphTopo::RenameEdge(Edge* edge, const string& new_name) {
    auto ret_pair = name2eid_.insert(make_pair(new_name, edge->GetId()));
    if (!ret_pair.second) {
        return false;
    }

    name2eid_.erase(edge->GetName());
    static_cast<FullGraphEdge*>(edge)->shared_name_str_ = &ret_pair.first->first;
    return true;
}

}}} // namespace ppl::nn::ir
