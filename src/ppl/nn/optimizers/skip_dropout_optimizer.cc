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

#include "ppl/nn/optimizers/skip_dropout_optimizer.h"

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


RetCode SkipDropoutOptimizer::Optimize(ir::Graph* graph) const {
    for(auto it = graph->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if(node->GetType().domain.empty() && node->GetType().name == "Dropout") {
            auto input_edge = graph->topo->GetEdgeById(node->GetInput(0));
            if(input_edge->CalcConsumerCount() != 1 || IsGraphOutput(graph, input_edge->GetId())) {
                continue;
            }
            auto output_edge = graph->topo->GetEdgeById(node->GetOutput(0));
            auto node_pre = graph->topo->GetNodeById(input_edge->GetProducer());	
            node_pre->ReplaceOutput(input_edge->GetId(), output_edge->GetId());
            output_edge->SetProducer(node_pre->GetId());

            graph->topo->DelEdgeById(input_edge->GetId());
            graph->topo->DelEdgeById(node->GetOutput(1));
            graph->topo->DelNodeById(node->GetId());
        }
    }

    return RC_SUCCESS;
}


}}