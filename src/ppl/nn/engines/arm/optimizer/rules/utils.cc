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

#include "ppl/nn/engines/arm/optimizer/rules/utils.h"
#include "ppl/nn/common/logger.h"

namespace ppl { namespace nn { namespace arm {

// create & register arm::OptKernel from ir::Node
ppl::common::RetCode CreateArmOptKernel(const OptKernelOptions& options, const ir::Node* node, ArmOptKernel** kernel) {
    auto& type = node->GetType();

    auto creator = OptKernelCreatorManager::GetInstance()->Find(type.domain, type.name, type.version);
    if (!creator) {
        LOG(ERROR) << "cannot find creator for ArmOptKernel[" << node->GetName() << "] type[" << type.domain << ":"
                   << type.name << "]";
        return ppl::common::RC_NOT_FOUND;
    }

    auto opt_kernel = std::unique_ptr<ArmOptKernel>((*creator)(node));
    if (!opt_kernel) {
        LOG(ERROR) << "create ArmOptKernel failed: oom";
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    auto status = opt_kernel->Init(options);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "Init for kernel[" << node->GetName() << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    *kernel = opt_kernel.get();
    options.info->kernels.emplace(node->GetId(), std::move(opt_kernel));

    return ppl::common::RC_SUCCESS;
}

// get inner edges from a sequence of nodes. sequence_nodes must be in order
std::set<ir::Edge*> GetSequenceNodesInnerEdges(ir::GraphTopo* graph_topo, std::vector<ir::Node*>& sequence_nodes) {
    std::set<ir::Edge*> inner_edges;
    if (sequence_nodes.empty()) {
        return inner_edges;
    }

    auto start_node = sequence_nodes[0];
    auto end_node = sequence_nodes[sequence_nodes.size() - 1];
    for (auto node : sequence_nodes) {
        for (uint32_t i = 0; i < node->GetInputCount(); i++) {
            auto input_edge = graph_topo->GetEdge(node->GetInput(i));
            if (!IsNodeInputEdge(start_node, input_edge)) {
                inner_edges.insert(input_edge);
            }
        }
        for (uint32_t i = 0; i < node->GetOutputCount(); i++) {
            auto output_edge = graph_topo->GetEdge(node->GetOutput(i));
            if (!IsNodeOutputEdge(end_node, output_edge)) {
                inner_edges.insert(output_edge);
            }
        }
    }

    return inner_edges;
}

// replace a sequence of nodes with one single node. sequence_nodes must be in order
ppl::common::RetCode ReplaceSequenceNodes(const OptKernelOptions& options, std::vector<ir::Node*>& sequence_nodes,
                                          ir::Node* target_node) {
    if (sequence_nodes.empty()) {
        LOG(ERROR) << "sequence nodes empty";
        return ppl::common::RC_INVALID_VALUE;
    }
    if (target_node == nullptr) {
        LOG(ERROR) << "target node empty";
        return ppl::common::RC_INVALID_VALUE;
    }
    for (auto node : sequence_nodes) {
        if (node == target_node) {
            return ppl::common::RC_INVALID_VALUE;
        }
    }

    auto graph_topo = options.graph_topo;
    auto graph_data = options.graph_data;
    auto info = options.info;
    auto& tensors = *options.tensors;

    // connect target_node to graph
    auto start_node = sequence_nodes[0];
    auto end_node = sequence_nodes[sequence_nodes.size() - 1];
    const auto input_edges = GetInputEdges(graph_topo, start_node);
    const auto output_edges = GetOutputEdges(graph_topo, end_node);

    for (auto input_edge : input_edges) {
        target_node->AddInput(input_edge->GetId());
        input_edge->DelConsumer(start_node->GetId());
        input_edge->AddConsumer(target_node->GetId());
    }
    for (auto output_edge : output_edges) {
        target_node->AddOutput(output_edge->GetId());
        output_edge->SetProducer(target_node->GetId());
    }

    // delete inner edges
    auto inner_edges = GetSequenceNodesInnerEdges(graph_topo, sequence_nodes);
    for (auto inner_edge : inner_edges) {
        for (auto node : sequence_nodes) {
            inner_edge->DelConsumer(node->GetId());
        }
        if (inner_edge->CalcConsumerCount() == 0 &&
            graph_data->constants.find(inner_edge->GetId()) == graph_data->constants.end()) {
            tensors.erase(inner_edge->GetId());
            graph_topo->DelEdge(inner_edge->GetId());
        }
    }

    // delete nodes
    for (auto node : sequence_nodes) {
        info->kernels.erase(node->GetId());
        graph_topo->DelNode(node->GetId());
    }

    return ppl::common::RC_SUCCESS;
}

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
            inner_edges.insert(graph_topo->GetEdge(node->GetInput(i)));
        }
        for (uint32_t i = 0; i < node->GetOutputCount(); i++) {
            inner_edges.insert(graph_topo->GetEdge(node->GetOutput(i)));
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
            graph_topo->DelEdge(inner_edge->GetId());
        }
    }

    // delete nodes
    for (auto node : nodes) {
        info->kernels.erase(node->GetId());
        graph_topo->DelNode(node->GetId());
    }

    return ppl::common::RC_SUCCESS;
}

bool IsReLU6(const ir::GraphTopo* graph_topo, const ir::GraphData* graph_data, const ir::Node* node) {
    if (node->GetType().domain != "" || node->GetType().name != "Clip") {
        return false;
    }
    if (node->GetInputCount() != 3) {
        return false;
    }
    auto clip_node = node;

    auto& constants = graph_data->constants;
    auto& shapes = graph_data->shapes;

    auto min_edge_id = clip_node->GetInput(1);
    auto max_edge_id = clip_node->GetInput(2);
    auto min_constant_it = constants.find(min_edge_id);
    auto max_constant_it = constants.find(max_edge_id);

    if (min_constant_it == constants.end()) { // maybe insert a reorder op between initializer and clip
        auto min_edge = graph_topo->GetEdge(min_edge_id);
        if (min_edge == nullptr) {
            return false;
        }
        auto reorder_node = graph_topo->GetNode(min_edge->GetProducer());
        if (reorder_node == nullptr || reorder_node->GetType().domain != "pmx" ||
            reorder_node->GetType().name != "Reorder") {
            return false;
        }
        auto reorder_input = graph_topo->GetEdge(reorder_node->GetInput(0));
        if (reorder_input == nullptr) {
            return false;
        }
        min_edge = reorder_input;
        min_edge_id = reorder_input->GetId();
        min_constant_it = constants.find(reorder_input->GetId());
    }
    if (max_constant_it == constants.end()) { // maybe insert a reorder op between initializer and clip
        auto max_edge = graph_topo->GetEdge(max_edge_id);
        if (max_edge == nullptr) {
            return false;
        }
        auto reorder_node = graph_topo->GetNode(max_edge->GetProducer());
        if (reorder_node == nullptr || reorder_node->GetType().domain != "pmx" ||
            reorder_node->GetType().name != "Reorder") {
            return false;
        }
        auto reorder_input = graph_topo->GetEdge(reorder_node->GetInput(0));
        if (reorder_input == nullptr) {
            return false;
        }
        max_edge = reorder_input;
        max_edge_id = reorder_input->GetId();
        max_constant_it = constants.find(reorder_input->GetId());
        if (max_constant_it == constants.end()) {
            return false;
        }
    }

    if (min_constant_it == constants.end() || max_constant_it == constants.end()) {
        return false;
    }

    auto min_edge_shape = shapes.find(min_edge_id);
    auto max_edge_shape = shapes.find(max_edge_id);
    if (min_edge_shape == shapes.end() || max_edge_shape == shapes.end()) {
        return false;
    }
    if (min_edge_shape->second.data_type != ppl::common::DATATYPE_FLOAT32 ||
        max_edge_shape->second.data_type != ppl::common::DATATYPE_FLOAT32) {
        return false;
    }

    float min_val = *((float*)min_constant_it->second.data.GetData());
    float max_val = *((float*)max_constant_it->second.data.GetData());
    if (min_val != 0.0f && max_val != 6.0f) {
        return false;
    }

    return true;
}

ppl::common::RetCode DelActivationNode(const OptKernelOptions& options, ir::Node* node) {
    if (node == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }

    auto graph_topo = options.graph_topo;
    auto graph_data = options.graph_data;
    auto info = options.info;
    auto& tensors = *options.tensors;

    auto input_edge = graph_topo->GetEdge(node->GetInput(0));
    auto predecessor_node = graph_topo->GetNode(input_edge->GetProducer());
    if (predecessor_node == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }

    auto output_edge = graph_topo->GetEdge(node->GetOutput(0));
    if (output_edge == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }

    predecessor_node->ReplaceOutput(node->GetInput(0), node->GetOutput(0));
    output_edge->SetProducer(predecessor_node->GetId());

    for (uint32_t i = 0; i < node->GetInputCount(); i++) {
        auto input_edge = graph_topo->GetEdge(node->GetInput(i));
        if (input_edge) {
            input_edge->DelConsumer(node->GetId());
        }
        if (input_edge->CalcConsumerCount() == 0 &&
            graph_data->constants.find(input_edge->GetId()) == graph_data->constants.end() &&
            !IsGraphInput(graph_topo, input_edge) && !IsGraphOutput(graph_topo, input_edge)) {
            tensors.erase(input_edge->GetId());
            graph_topo->DelEdge(input_edge->GetId());
        }
    }

    info->kernels.erase(node->GetId());
    graph_topo->DelNode(node->GetId());

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::arm
