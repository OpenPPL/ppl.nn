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

#include <vector>

#include "ppl/nn/engines/arm/optimizer/rules/fuse_conv_activation.h"
#include "ppl/nn/engines/arm/optimizer/rules/utils.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/conv_op.h"

namespace ppl { namespace nn { namespace arm {

bool FuseConvActivationRule::ApplySingleInputOutputActivationNode(const OptKernelOptions& options) {
    bool graph_changed = false;

    auto graph_topo = options.graph_topo;
    auto info = options.info;

    for (auto it = graph_topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "Conv") {
            auto conv_node = node;
            auto conv_output_edge_id = conv_node->GetOutput(0);
            auto conv_output_edge = graph_topo->GetEdge(conv_output_edge_id);
            if (conv_output_edge->CalcConsumerCount() != 1) {
                continue;
            }
            if (IsGraphInput(graph_topo, conv_output_edge_id) || IsGraphOutput(graph_topo, conv_output_edge_id)) {
                continue;
            }
            auto conv_kernel = static_cast<ConvOp*>(info->kernels[conv_node->GetId()].get());

            auto successor_node_id = conv_output_edge->CreateConsumerIter().Get();
            auto successor_node = graph_topo->GetNode(successor_node_id);
            if (successor_node->GetType().domain != "") {
                continue;
            }
            if (successor_node->GetType().name == "Relu") {
                if (!conv_kernel->TryFuseReLU()) { // set fuse flag to conv_op
                    continue;
                }
            } else {
                continue;
            }

            LOG(INFO) << "merge kernel " << successor_node->GetName() << " into kernel " << conv_node->GetName() << ".";
            DelActivationNode(options, successor_node);
            graph_changed = true;
        }
    }

    return graph_changed;
}

bool FuseConvActivationRule::ApplyReLU6(const OptKernelOptions& options) {
    bool graph_changed = false;

    auto graph_topo = options.graph_topo;
    auto graph_data = options.graph_data;
    auto info = options.info;
    auto& tensors = *options.tensors;

    for (auto it = graph_topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "Conv") {
            auto conv_node = node;
            auto conv_output_edge_id = conv_node->GetOutput(0);
            auto conv_output_edge = graph_topo->GetEdge(conv_output_edge_id);
            if (conv_output_edge->CalcConsumerCount() != 1) {
                continue;
            }
            if (IsGraphInput(graph_topo, conv_output_edge_id) || IsGraphOutput(graph_topo, conv_output_edge_id)) {
                continue;
            }
            auto conv_kernel = static_cast<ConvOp*>(info->kernels[conv_node->GetId()].get());

            auto successor_node_id = conv_output_edge->CreateConsumerIter().Get();
            auto successor_node = graph_topo->GetNode(successor_node_id);
            if (!IsReLU6(graph_topo, graph_data, successor_node)) {
                continue;
            }
            if (!conv_kernel->TryFuseReLU6()) { // set fuse flag to conv_op
                continue;
            }

            // del min/max edge and relational reorder op
            auto min_edge = graph_topo->GetEdge(successor_node->GetInput(1));
            auto max_edge = graph_topo->GetEdge(successor_node->GetInput(2));
            min_edge->DelConsumer(successor_node->GetId());
            max_edge->DelConsumer(successor_node->GetId());
            if (min_edge->CalcConsumerCount() == 0 && !IsGraphOutput(graph_topo, min_edge->GetId())) {
                if (graph_data->constants.find(min_edge->GetId()) != graph_data->constants.end()) {
                    tensors.erase(min_edge->GetId());
                    graph_data->constants.erase(min_edge->GetId());
                    graph_topo->DelEdge(min_edge->GetId());
                } else { // has an reorder op between initializer & clip
                    auto reorder_node = graph_topo->GetNode(min_edge->GetProducer());
                    auto min_initializer_edge = graph_topo->GetEdge(reorder_node->GetInput(0));

                    tensors.erase(min_edge->GetId());
                    graph_topo->DelEdge(min_edge->GetId());

                    min_initializer_edge->DelConsumer(reorder_node->GetId());
                    info->kernels.erase(reorder_node->GetId());
                    graph_topo->DelNode(reorder_node->GetId());

                    if (min_initializer_edge->CalcConsumerCount() == 0 &&
                        !IsGraphOutput(graph_topo, min_initializer_edge->GetId())) {
                        tensors.erase(min_initializer_edge->GetId());
                        graph_data->constants.erase(min_initializer_edge->GetId());
                        graph_topo->DelEdge(min_initializer_edge->GetId());
                    }
                }
            }
            if (max_edge->CalcConsumerCount() == 0 && !IsGraphOutput(graph_topo, max_edge->GetId())) {
                if (graph_data->constants.find(max_edge->GetId()) != graph_data->constants.end()) {
                    tensors.erase(max_edge->GetId());
                    graph_data->constants.erase(max_edge->GetId());
                    graph_topo->DelEdge(max_edge->GetId());
                } else { // has an reorder op between initializer & clip
                    auto reorder_node = graph_topo->GetNode(max_edge->GetProducer());
                    auto max_initializer_edge = graph_topo->GetEdge(reorder_node->GetInput(0));

                    tensors.erase(max_edge->GetId());
                    graph_topo->DelEdge(max_edge->GetId());

                    max_initializer_edge->DelConsumer(reorder_node->GetId());
                    info->kernels.erase(reorder_node->GetId());
                    graph_topo->DelNode(reorder_node->GetId());

                    if (max_initializer_edge->CalcConsumerCount() == 0 &&
                        !IsGraphOutput(graph_topo, max_initializer_edge->GetId())) {
                        tensors.erase(max_initializer_edge->GetId());
                        graph_data->constants.erase(max_initializer_edge->GetId());
                        graph_topo->DelEdge(max_initializer_edge->GetId());
                    }
                }
            }

            auto activation_node = successor_node;
            auto activation_node_id = activation_node->GetId();
            auto activation_output_edge_id = activation_node->GetOutput(0);
            auto activation_output_edge = graph_topo->GetEdge(activation_output_edge_id);
            // conv_node -> conv_output_edge -> activation_node -> activation_output_edge
            // conv_node                                        -> activation_output_edge
            conv_node->ReplaceOutput(conv_output_edge_id, activation_output_edge_id);
            activation_output_edge->SetProducer(conv_node->GetId());

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
            LOG(INFO) << "merge kernel " << successor_node->GetName() << " into kernel " << conv_node->GetName() << ".";
#endif
            info->kernels.erase(activation_node_id);
            graph_topo->DelNode(activation_node_id);
            graph_topo->DelEdge(conv_output_edge_id);
            graph_changed = true;
        }
    }

    return graph_changed;
}

bool FuseConvActivationRule::Apply(const OptKernelOptions& options) {
    return ApplySingleInputOutputActivationNode(options) || ApplyReLU6(options);
}

}}} // namespace ppl::nn::arm
