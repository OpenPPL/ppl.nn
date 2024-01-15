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
#include "ppl/nn/common/logger.h"

using namespace ppl::common;

namespace ppl { namespace nn {

static void DeleteNodeAndOutput(const ir::Node* node, ir::GraphTopo* topo) {
    auto in_eid = node->GetInput(0);
    auto out_eid = node->GetOutput(0);
    auto in_edge = topo->GetEdge(in_eid);
    auto out_edge = topo->GetEdge(out_eid);

    for (uint32_t i = 0; i < node->GetInputCount(); ++i) {
        auto edge = topo->GetEdge(node->GetInput(i));
        edge->DelConsumer(node->GetId());
    }

    for (auto it = out_edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
        auto nid = it.Get();
        auto next = topo->GetNode(nid);
        auto nr = next->ReplaceInput(out_eid, in_eid);
        if (nr > 0) {
            in_edge->AddConsumer(nid);
        }
    }

    topo->DelEdge(out_eid);
    topo->DelEdge(node->GetOutput(1));
    topo->DelNode(node->GetId());
}

static void DeleteNodeAndInput(const ir::Node* node, ir::GraphTopo* topo) {
    auto in_eid = node->GetInput(0);
    auto out_eid = node->GetOutput(0);
    auto in_edge = topo->GetEdge(in_eid);
    auto out_edge = topo->GetEdge(out_eid);

    for (uint32_t i = 1; i < node->GetInputCount(); ++i) {
        auto edge = topo->GetEdge(node->GetInput(i));
        edge->DelConsumer(node->GetId());
    }

    auto prev = topo->GetNode(in_edge->GetProducer());
    auto nr = prev->ReplaceOutput(in_eid, out_eid);
    if (nr > 0) {
        out_edge->SetProducer(prev->GetId());
    }

    topo->DelEdge(in_eid);
    topo->DelEdge(node->GetOutput(1));
    topo->DelNode(node->GetId());
}

static bool IsGraphInput(const ir::GraphTopo* topo, edgeid_t edge_id) {
    for (uint32_t i = 0; i < topo->GetInputCount(); i++) {
        if (topo->GetInput(i) == edge_id) {
            return true;
        }
    }
    return false;
}

static bool IsGraphOutput(const ir::GraphTopo* topo, edgeid_t edge_id) {
    for (uint32_t i = 0; i < topo->GetOutputCount(); i++) {
        if (topo->GetOutput(i) == edge_id) {
            return true;
        }
    }
    return false;
}

RetCode SkipDropoutOptimizer::Optimize(ir::Graph* graph) const {
    auto topo = graph->topo.get();
    for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        auto& type = node->GetType();
        if (type.name == "Dropout") {
            if (node->GetOutputCount() > 2) {
                LOG(ERROR) << "unsupported Dropout version [" << type.version << "]";
                return RC_UNSUPPORTED;
            }

            bool input_is_graph_input = IsGraphInput(topo, node->GetInput(0));
            bool output_is_graph_output = IsGraphOutput(topo, node->GetOutput(0));

            if (input_is_graph_input && output_is_graph_output) {
                continue;
            }

            if (node->GetOutputCount() == 2) {
                auto mask_edge = topo->GetEdge(node->GetOutput(1));
                if (mask_edge->CalcConsumerCount() > 0) {
                    continue;
                }
            }

            if (output_is_graph_output) {
                DeleteNodeAndInput(node, topo);
            } else {
                DeleteNodeAndOutput(node, topo);
            }
        }
    }

    return RC_SUCCESS;
}

}} // namespace ppl::nn
