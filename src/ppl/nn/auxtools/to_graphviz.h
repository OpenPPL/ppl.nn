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

static string GenNodeIdStr(const ir::Node* node) {
    auto& type = node->GetType();
    return node->GetName() + "[" + type.domain + ":" + type.name + ":" + std::to_string(type.version) + "]";
}

static string ToGraphviz(const ir::GraphTopo* topo) {
    string content = "digraph NetGraph {\n";

    for (auto it = topo->CreateEdgeIter(); it->IsValid(); it->Forward()) {
        auto edge = it->Get();

        string begin_node_name;
        if (edge->GetProducer() == INVALID_NODEID) {
            begin_node_name = "NIL-BEGIN";
        } else {
            auto node = topo->GetNodeById(edge->GetProducer());
            begin_node_name = GenNodeIdStr(node);
        }

        auto consumer_iter = edge->CreateConsumerIter();
        if (consumer_iter.IsValid()) {
            do {
                auto node = topo->GetNodeById(consumer_iter.Get());
                content += "\"" + begin_node_name + "\" -> \"" + GenNodeIdStr(node) + "\" [label=\"" + edge->GetName() +
                    "\"]\n";
                consumer_iter.Forward();
            } while (consumer_iter.IsValid());
        } else {
            content += "\"" + begin_node_name + "\" -> \"NIL-END\" [label=\"" + edge->GetName() + "\"]\n";
            consumer_iter.Forward();
        }
    }

    content += "}";
    return content;
}

}}} // namespace ppl::nn::utils

#endif
