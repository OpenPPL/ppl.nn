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

#include "ppl/nn/optimizers/identity_node_optimizer.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

inline bool IsGraphOutput(const ir::Graph* graph, edgeid_t edge_id) {
    for (uint32_t i = 0; i < graph->topo->GetOutputCount(); i++) {
        if (graph->topo->GetOutput(i) == edge_id) {
            return true;
        }
    }
    return false;
}

inline bool IsGraphInput(const ir::Graph* graph, edgeid_t edge_id) {
    for (uint32_t i = 0; i < graph->topo->GetInputCount(); i++) {
        if (graph->topo->GetInput(i) == edge_id) {
            return true;
        }
    }
    return false;
}

inline bool IsConstant(const ir::Graph* graph, edgeid_t edge_id) {
    return graph->data->constants.find(edge_id) != graph->data->constants.end();
}

// move identity op's output edge to graph identity & delete identity op
ppl::common::RetCode IdentityNodeOptimizer::Optimize(ir::Graph* graph) const {
    for (auto it = graph->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain.empty() && node->GetType().name == "Identity") {
            auto identity_node = node;
            auto in_edge = graph->topo->GetEdge(node->GetInput(0));
            auto out_edge = graph->topo->GetEdge(node->GetOutput(0));
            if (!in_edge || !out_edge) {
                LOG(ERROR) << "cannot find identity node[" << identity_node->GetName() << "]'s input/output edge.";
                return RC_NOT_FOUND;
            }

            const bool in_edge_is_input = IsGraphInput(graph, in_edge->GetId());
            const bool in_edge_is_constant = IsConstant(graph, in_edge->GetId());
            const bool out_edge_is_output = IsGraphOutput(graph, out_edge->GetId());
            if ((in_edge_is_input || in_edge_is_constant) && out_edge_is_output) {
                continue;
            }

            if (in_edge_is_input || in_edge_is_constant) {
                in_edge->DelConsumer(identity_node->GetId());
                graph->topo->DelNode(identity_node->GetId());

                auto iter = out_edge->CreateConsumerIter();
                while (iter.IsValid()) {
                    auto next_node_id = iter.Get();
                    auto next_node = graph->topo->GetNode(next_node_id);
                    next_node->ReplaceInput(out_edge->GetId(), in_edge->GetId());
                    in_edge->AddConsumer(next_node_id);
                    iter.Forward();
                }

                graph->topo->DelEdge(out_edge->GetId());
            } else {
                auto prev_node_id = in_edge->GetProducer();
                auto prev_node = graph->topo->GetNode(prev_node_id);

                prev_node->ReplaceOutput(in_edge->GetId(), out_edge->GetId());
                out_edge->SetProducer(prev_node_id);

                graph->topo->DelEdge(in_edge->GetId());
                graph->topo->DelNode(identity_node->GetId());
            }
        }
    }

    return RC_SUCCESS;
}

}} // namespace ppl::nn
