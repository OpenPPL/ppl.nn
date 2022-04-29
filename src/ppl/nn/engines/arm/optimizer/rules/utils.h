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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_RULES_UTILS_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_RULES_UTILS_H_

#include <vector>
#include <set>

#include "ppl/nn/ir/graph.h"
#include "ppl/nn/engines/arm/optimizer/opt_kernel_creator_manager.h"
#include "ppl/common/retcode.h"

namespace ppl { namespace nn { namespace arm {

inline bool IsGraphOutput(const ir::GraphTopo* graph_topo, edgeid_t edge_id) {
    for (uint32_t i = 0; i < graph_topo->GetOutputCount(); i++) {
        if (graph_topo->GetOutput(i) == edge_id) {
            return true;
        }
    }
    return false;
}

inline bool IsGraphOutput(const ir::GraphTopo* graph_topo, ir::Edge* edge) {
    if (edge) {
        return IsGraphOutput(graph_topo, edge->GetId());
    }
    return false;
}

inline bool IsGraphInput(const ir::GraphTopo* graph_topo, edgeid_t edge_id) {
    for (uint32_t i = 0; i < graph_topo->GetInputCount(); i++) {
        if (graph_topo->GetInput(i) == edge_id) {
            return true;
        }
    }
    return false;
}

inline bool IsGraphInput(const ir::GraphTopo* graph_topo, ir::Edge* edge) {
    if (edge) {
        return IsGraphInput(graph_topo, edge->GetId());
    }
    return false;
}

inline std::vector<ir::Edge*> GetInputEdges(ir::GraphTopo* graph_topo, ir::Node* node) {
    std::vector<ir::Edge*> input_edges(node->GetInputCount(), nullptr);
    for (uint32_t i = 0; i < node->GetInputCount(); i++) {
        input_edges[i] = graph_topo->GetEdge(node->GetInput(i));
    }
    return input_edges;
}

inline std::vector<ir::Edge*> GetOutputEdges(ir::GraphTopo* graph_topo, ir::Node* node) {
    std::vector<ir::Edge*> output_edges(node->GetOutputCount(), nullptr);
    for (uint32_t i = 0; i < node->GetOutputCount(); i++) {
        output_edges[i] = graph_topo->GetEdge(node->GetOutput(i));
    }
    return output_edges;
}

inline bool IsNodeInputEdge(const ir::Node* node, const ir::Edge* edge) {
    for (uint32_t i = 0; i < node->GetInputCount(); i++) {
        if (edge->GetId() == node->GetInput(i)) {
            return true;
        }
    }
    return false;
}

inline bool IsNodeOutputEdge(const ir::Node* node, const ir::Edge* edge) {
    for (uint32_t i = 0; i < node->GetOutputCount(); i++) {
        if (edge->GetId() == node->GetOutput(i)) {
            return true;
        }
    }
    return false;
}

// create & register arm::OptKernel from ir::Node
ppl::common::RetCode CreateArmOptKernel(const OptKernelOptions& options, const ir::Node* node, ArmOptKernel** kernel);

// get inner edges from a sequence of nodes. sequence_nodes must be in order
std::set<ir::Edge*> GetSequenceNodesInnerEdges(ir::GraphTopo* graph_topo, std::vector<ir::Node*>& sequence_nodes);

// replace a sequence of nodes with one single node. sequence_nodes must be in order
ppl::common::RetCode ReplaceSequenceNodes(const OptKernelOptions& options, std::vector<ir::Node*>& sequence_nodes,
                                          ir::Node* target_node);

// replace subgraph with one node
ppl::common::RetCode ReplaceSubgraphWithOneNode(const OptKernelOptions& options, std::vector<ir::Node*>& nodes,
                                                std::vector<ir::Edge*>& inputs, std::vector<ir::Edge*>& outputs,
                                                ir::Node* target_node);

// whether a node is ReLU6
bool IsReLU6(const ir::GraphTopo* graph_topo, const ir::GraphData* graph_data, const ir::Node* node);

// del activation node
ppl::common::RetCode DelActivationNode(const OptKernelOptions& options, ir::Node* node);

}}} // namespace ppl::nn::arm

#endif
