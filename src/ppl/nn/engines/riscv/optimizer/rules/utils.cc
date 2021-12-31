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

#include "ppl/nn/engines/riscv/optimizer/rules/utils.h"

namespace ppl { namespace nn { namespace riscv {

// replace subgraph with one node
ppl::common::RetCode ReplaceSubgraphWithOneNode(const OptKernelOptions& options, std::vector<ir::Node*>& nodes,
                                                std::vector<ir::Edge*>& inputs, std::vector<ir::Edge*>& outputs,
                                                ir::Node* target_node) {
    if (nodes.empty() || inputs.empty() || outputs.empty()) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (target_node == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    for (auto input : inputs) {
        if (input == nullptr) {
            return ppl::common::RC_INVALID_VALUE;
        }
    }
    for (auto output : outputs) {
        if (output == nullptr) {
            return ppl::common::RC_INVALID_VALUE;
        }
    }
    for (auto node : nodes) {
        if (node == target_node || node == nullptr) {
            return ppl::common::RC_INVALID_VALUE;
        }
    }

    auto graph_topo = options.graph_topo;
    auto graph_data = options.graph_data;
    auto info = options.info;
    auto& tensors = *options.tensors;

    // get inner edges
    std::set<ir::Edge*> inner_edges;
    for (auto node : nodes) {
        for (uint32_t i = 0; i < node->GetInputCount(); i++) {
            inner_edges.insert(graph_topo->GetEdgeById(node->GetInput(i)));
        }
        for (uint32_t i = 0; i < node->GetOutputCount(); i++) {
            inner_edges.insert(graph_topo->GetEdgeById(node->GetOutput(i)));
        }
    }
    for (auto input : inputs) {
        if (inner_edges.find(input) == inner_edges.end()) {
            return ppl::common::RC_INVALID_VALUE;
        }
        inner_edges.erase(input);
    }
    for (auto output : outputs) {
        if (inner_edges.find(output) == inner_edges.end()) {
            return ppl::common::RC_INVALID_VALUE;
        }
        inner_edges.erase(output);
    }

    // connect target_node to graph
    for (auto input : inputs) {
        target_node->AddInput(input->GetId());
        input->AddConsumer(target_node->GetId());
        for (auto node : nodes) {
            input->DelConsumer(node->GetId());
        }
    }
    for (auto output : outputs) {
        target_node->AddOutput(output->GetId());
        output->SetProducer(target_node->GetId());
    }

    // delete inner edges
    for (auto inner_edge : inner_edges) {
        for (auto node : nodes) {
            inner_edge->DelConsumer(node->GetId());
        }
        if (inner_edge->CalcConsumerCount() == 0 &&
            graph_data->constants.find(inner_edge->GetId()) == graph_data->constants.end()) {
            tensors.erase(inner_edge->GetId());
            graph_topo->DelEdgeById(inner_edge->GetId());
        }
    }

    // delete nodes
    for (auto node : nodes) {
        info->kernels.erase(node->GetId());
        graph_topo->DelNodeById(node->GetId());
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::riscv