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

#ifndef _ST_HPC_PPL_NN_AUXTOOLS_TO_GRAPHVIZ_H_
#define _ST_HPC_PPL_NN_AUXTOOLS_TO_GRAPHVIZ_H_

#include <string>
#include "ppl/nn/ir/graph_topo.h"
using namespace std;

namespace ppl { namespace nn { namespace utils {

static inline string GetGraphvizNodeName(const ir::Node* node) {
    string type_str;
    if (node->GetType().domain.empty()) {
        type_str = node->GetType().name;
    } else {
        type_str = node->GetType().domain + "." + node->GetType().name;
    }
    return node->GetName() + "[" + type_str + "][" + std::to_string(node->GetId()) + "]";
}

static string ToGraphviz(const ir::GraphTopo* topo) {
    string content = "digraph NetGraph {\n";

    for (auto it = topo->CreateEdgeIter(); it->IsValid(); it->Forward()) {
        auto edge = it->Get();

        string begin_node_name;
        if (edge->GetProducer() == INVALID_NODEID) {
            if (topo->GetInput(edge->GetName()) != INVALID_EDGEID) {
                begin_node_name = "input:" + edge->GetName();
            } else if (topo->GetExtraInput(edge->GetName()) != INVALID_EDGEID) {
                begin_node_name = "extra_input:" + edge->GetName();
            }
        } else {
            auto node = topo->GetNodeById(edge->GetProducer());
            begin_node_name = GetGraphvizNodeName(node);
        }

        auto edge_iter = edge->CreateConsumerIter();
        if (edge_iter.IsValid()) {
            do {
                auto node = topo->GetNodeById(edge_iter.Get());
                if (!begin_node_name.empty()) {
                    content += "\"" + begin_node_name + "\" -> \"" + GetGraphvizNodeName(node) + "\" [label=\"" +
                        edge->GetName() + "\"]\n";
                }
                edge_iter.Forward();
            } while (edge_iter.IsValid());
        } else {
            if (!begin_node_name.empty()) {
                content += "\"" + begin_node_name + "\" -> \"output:" + edge->GetName() + "\" [label=\"" +
                    edge->GetName() + "\"]\n";
            }
            edge_iter.Forward();
        }
    }

    content += "}";
    return content;
}

}}} // namespace ppl::nn::utils

#endif
