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

#include "ppl/nn/engines/x86/optimizer/rules/fuse_conv_eltwise.h"
#include "ppl/nn/engines/x86/optimizer/rules/utils.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/conv_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/add_op.h"

namespace ppl { namespace nn { namespace x86 {

bool FuseConvEltwise(const OptKernelOptions &options) {
    bool graph_changed = false;
    auto graph_topo = options.graph_topo;
    auto info = options.info;
    auto &tensors = *options.tensors;

    for (auto it = graph_topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "Add") {
            auto add_node = node;
            auto input_edge_0 = graph_topo->GetEdgeById(node->GetInput(0));
            auto input_edge_1 = graph_topo->GetEdgeById(node->GetInput(1));

            // check if eltwise
            auto& input_shape_0 = tensors[input_edge_0->GetId()]->GetShape();
            auto& input_shape_1 = tensors[input_edge_1->GetId()]->GetShape();
            if (input_shape_0.IsEmpty() || input_shape_1.IsEmpty()) { // input shape has not been infered
                continue;
            }
            if (input_shape_0.GetDimCount() != input_shape_1.GetDimCount()) {
                continue;
            }
            bool same_dim = true;
            for (uint32_t i = 0; i < std::min(input_shape_0.GetRealDimCount(), input_shape_1.GetRealDimCount()); i++) {
                if (input_shape_0.GetDim(i) != input_shape_1.GetDim(i)) {
                    same_dim = false;
                    break;
                }
            }
            if (!same_dim) {
                continue;
            }

            ir::Node* conv_node = nullptr;
            ir::Edge* src_sum_edge = nullptr;
            auto add_op = (AddOp*)info->kernels[add_node->GetId()].get();
            if (!conv_node && input_edge_0->GetProducer() != INVALID_NODEID && input_edge_0->CalcConsumerCount() == 1 &&
                !IsGraphOutput(graph_topo, input_edge_0->GetId())) {
                auto predecessor_node_0 = graph_topo->GetNodeById(input_edge_0->GetProducer());
                if (predecessor_node_0->GetType().domain == "" && predecessor_node_0->GetType().name == "Conv") {
                    auto conv_op = (ConvOp*)info->kernels[predecessor_node_0->GetId()].get();
                    if (conv_op->TryFuseSum()) {
                        if (add_op->HasFuseReLU()) { // becareful
                            conv_op->TryFuseReLU();
                        }
                        conv_node = predecessor_node_0;
                        src_sum_edge = input_edge_1;
                    }
                }
            }

            if (!conv_node && input_edge_1->GetProducer() != INVALID_NODEID && input_edge_1->CalcConsumerCount() == 1 &&
                !IsGraphOutput(graph_topo, input_edge_1->GetId())) {
                auto predecessor_node_1 = graph_topo->GetNodeById(input_edge_1->GetProducer());
                if (predecessor_node_1->GetType().domain == "" && predecessor_node_1->GetType().name == "Conv") {
                    auto conv_op = (ConvOp*)info->kernels[predecessor_node_1->GetId()].get();
                    if (conv_op->TryFuseSum()) {
                        if (add_op->HasFuseReLU()) {
                            conv_op->TryFuseReLU();
                        }
                        conv_node = predecessor_node_1;
                        src_sum_edge = input_edge_0;
                    }
                }
            }

            if (!conv_node) {
                continue;
            }
            auto conv_output_edge_id = conv_node->GetOutput(0);
            auto add_output_edge = graph_topo->GetEdgeById(add_node->GetOutput(0));

            conv_node->AddInput(src_sum_edge->GetId()); // add src_sum_edge as input[-1] of conv_node
            src_sum_edge->AddConsumer(conv_node->GetId());
            src_sum_edge->DelConsumer(add_node->GetId());
            conv_node->ReplaceOutput(conv_output_edge_id, add_output_edge->GetId());
            add_output_edge->SetProducer(conv_node->GetId());

            // LOG(INFO) << "fuse add " << add_node->GetName() << ".";
            info->kernels.erase(add_node->GetId());
            graph_topo->DelNodeById(add_node->GetId());
            graph_topo->DelEdgeById(conv_output_edge_id);

            graph_changed = true;
        }
    }

    return graph_changed;
}

}}} // namespace ppl::nn::x86

