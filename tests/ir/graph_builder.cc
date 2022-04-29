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
#include "ppl/nn/common/logger.h"
#include "tests/ir/graph_builder.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace test {

GraphBuilder::GraphBuilder(const string& graph_name) {
    graph_.topo = make_shared<ir::FullGraphTopo>();
    graph_.data = make_shared<ir::GraphData>();
    graph_.topo->SetName(graph_name);
}

RetCode GraphBuilder::AddNode(const string& name, const ir::Node::Type& type, const vector<string>& inputs,
                              const vector<string>& outputs, const vector<string>& extra_inputs) {
    auto topo = graph_.topo.get();
    auto ret_pair = topo->AddNode(name);
    if (!ret_pair.second) {
        LOG(ERROR) << "node[" << name << "] already exists.";
        return RC_EXISTS;
    }
    auto node = ret_pair.first;

    node->SetType(type);

    for (auto x = inputs.begin(); x != inputs.end(); ++x) {
        auto edge_ret_pair = topo->AddEdge(*x);
        auto edge = edge_ret_pair.first;
        node->AddInput(edge->GetId());
        edge->AddConsumer(node->GetId());
    }

    for (auto x = outputs.begin(); x != outputs.end(); ++x) {
        auto edge_ret_pair = topo->AddEdge(*x);
        auto edge = edge_ret_pair.first;
        if (edge->GetProducer() != INVALID_NODEID) {
            LOG(ERROR) << "output[" << *x << "] already exists.";
            return RC_EXISTS;
        }
        node->AddOutput(edge->GetId());
        edge->SetProducer(node->GetId());
    }

    if (extra_inputs.size() > 0) {
        extra_inputs_[node->GetId()] = extra_inputs;
    }

    return RC_SUCCESS;
}

RetCode GraphBuilder::Finalize() {
    auto topo = graph_.topo.get();

    set<edgeid_t> extra_input_ids;
    for (auto x = extra_inputs_.begin(); x != extra_inputs_.end(); ++x) {
        auto node = graph_.topo->GetNode(x->first);
        for (auto y = x->second.begin(); y != x->second.end(); ++y) {
            auto edge = topo->GetEdge(*y);
            if (!edge) {
                auto ret_pair = topo->AddEdge(*y);
                edge = ret_pair.first;
                topo->MarkAsExtraInput(edge->GetId());
                extra_input_ids.insert(edge->GetId());
            }
            node->AddExtraInput(edge->GetId());
        }
    }
    for (auto it = topo->CreateEdgeIter(); it->IsValid(); it->Forward()) {
        auto edge = it->Get();
        if (edge->GetProducer() == INVALID_NODEID) {
            if (extra_input_ids.find(edge->GetId()) == extra_input_ids.end()) {
                topo->MarkAsInput(edge->GetId());
            }
        }
        if (edge->CalcConsumerCount() == 0) {
            topo->MarkAsOutput(edge->GetId());
        }
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::test
