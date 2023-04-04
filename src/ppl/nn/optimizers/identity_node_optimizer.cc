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
            
            if (out_edge->CalcConsumerCount() == 0) {
                continue;
            }
            auto next_node_id = out_edge->CreateConsumerIter().Get();
            auto next_node = graph->topo->GetNode(next_node_id);
            next_node->ReplaceInput(out_edge->GetId(), in_edge->GetId());

            // delete constant node

            in_edge->DelConsumer(identity_node->GetId());
            in_edge->AddConsumer(next_node_id);
            out_edge->DelConsumer(next_node_id);
            graph->topo->DelNode(identity_node->GetId());
            if (out_edge->CalcConsumerCount() == 0) {
                graph->topo->DelEdge(out_edge->GetId());
            }
        }
    }

    return RC_SUCCESS;
}

}} // namespace ppl::nn
