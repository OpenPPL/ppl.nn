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

#include "ppl/nn/engines/x86/optimizer/rules/fuse_conv_activation.h"
#include "ppl/nn/engines/x86/optimizer/rules/utils.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/conv_op.h"

namespace ppl { namespace nn { namespace x86 {

static bool IsReLU6(const ir::GraphData* graph_data, const ir::Node* clip_node) {
    if (clip_node->GetType().domain != "" || clip_node->GetType().name != "Clip") {
        return false;
    }
    if (clip_node->GetInputCount() != 3) {
        return false;
    }

    auto min_edge_id = clip_node->GetInput(1);
    auto max_edge_id = clip_node->GetInput(2);
    auto& constants = graph_data->constants;
    auto min_edge_constant = constants.find(min_edge_id);
    auto max_edge_constant = constants.find(max_edge_id);
    if (min_edge_constant == constants.end() || max_edge_constant == constants.end()) {
        return false;
    }

    auto& shapes = graph_data->shapes;
    auto min_edge_shape = shapes.find(min_edge_id);
    auto max_edge_shape = shapes.find(max_edge_id);
    if (min_edge_shape == shapes.end() || max_edge_shape == shapes.end()) {
        return false;
    }
    if (min_edge_shape->second.data_type != ppl::common::DATATYPE_FLOAT32 || max_edge_shape->second.data_type != ppl::common::DATATYPE_FLOAT32) {
        return false;
    }

    float min_val = *((float*)min_edge_constant->second.data.data());
    float max_val = *((float*)max_edge_constant->second.data.data());
    if (min_val != 0.0f && max_val != 6.0f) {
        return false;
    }

    return true;
}

bool FuseConvActivation(const OptKernelOptions &options) {
    bool graphchanged = false;
    auto graph_topo = options.graph_topo;
    auto graph_data = options.graph_data;
    auto info = options.info;

    for (auto it = graph_topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "Conv") {
            auto conv_node = node;
            auto conv_output_edge_id = conv_node->GetOutput(0);
            auto conv_output_edge = graph_topo->GetEdgeById(conv_output_edge_id);
            if (conv_output_edge->CalcConsumerCount() != 1) {
                continue;
            }
            if (IsGraphOutput(graph_topo, conv_output_edge_id)) {
                continue;
            }

            auto successor_node_id = conv_output_edge->CreateConsumerIter().Get();
            auto successor_node = graph_topo->GetNodeById(successor_node_id);
            if (successor_node->GetType().domain != "") {
                continue;
            }

            auto conv_kernel = static_cast<ConvOp*>(info->kernels[conv_node->GetId()].get());
            if (successor_node->GetType().name == "Relu") {
                if (!conv_kernel->TryFuseReLU()) { // set fuse flag to conv_op
                    continue;
                }
            } else if (IsReLU6(graph_data, successor_node)) {
                if (!conv_kernel->TryFuseReLU6()) { // set fuse flag to conv_op
                    continue;
                }
                // remove relu6's input min/max's connect in advance
                auto min_edge = graph_topo->GetEdgeById(successor_node->GetInput(1));
                auto max_edge = graph_topo->GetEdgeById(successor_node->GetInput(2));
                min_edge->DelConsumer(successor_node->GetId());
                max_edge->DelConsumer(successor_node->GetId());
                if (min_edge->CalcConsumerCount() == 0 && !IsGraphOutput(graph_topo, min_edge->GetId())) {
                    graph_data->constants.erase(min_edge->GetId());
                    graph_topo->DelEdgeById(min_edge->GetId());
                }
                if (max_edge->CalcConsumerCount() == 0 && !IsGraphOutput(graph_topo, max_edge->GetId())) {
                    graph_data->constants.erase(max_edge->GetId());
                    graph_topo->DelEdgeById(max_edge->GetId());
                }
            } else {
                continue;
            }

            auto activation_node = successor_node;
            auto activation_node_id = activation_node->GetId();
            auto activation_output_edge_id = activation_node->GetOutput(0);
            auto activation_output_edge = graph_topo->GetEdgeById(activation_output_edge_id);
            // conv_node -> conv_output_edge -> activation_node -> activation_output_edge
            // conv_node                                        -> activation_output_edge
            conv_node->ReplaceOutput(conv_output_edge_id, activation_output_edge_id);
            activation_output_edge->SetProducer(conv_node->GetId());

            // LOG(INFO) << "merge kernel " << activation_node->GetName() << " into kernel " << conv_node->GetName() << ".";
            info->kernels.erase(activation_node_id);
            graph_topo->DelNodeById(activation_node_id);
            graph_topo->DelEdgeById(conv_output_edge_id);

            graphchanged = true;
        }
    }

    return graphchanged;
}

}}} // namespace ppl::nn::x86

