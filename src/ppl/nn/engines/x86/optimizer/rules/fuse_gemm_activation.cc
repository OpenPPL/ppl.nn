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

#include "ppl/nn/engines/x86/optimizer/rules/fuse_arithmetic_relu.h"
#include "ppl/nn/engines/x86/optimizer/rules/utils.h"
#include "ppl/nn/engines/x86/optimizer/opt_rule_manager.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/gemm_op.h"

namespace ppl { namespace nn { namespace x86 {

bool FuseGemmActivation(const OptKernelOptions &options) {
    bool graph_changed = false;
    auto graph_topo = options.graph_topo;
    auto info = options.info;
    auto &tensors = *options.tensors;

    for (auto it = graph_topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "Gemm") {
            auto gemm_node = node;
            auto gemm_output_edge_id = gemm_node->GetOutput(0);
            auto gemm_output_edge = graph_topo->GetEdgeById(gemm_output_edge_id);
            if (gemm_output_edge->CalcConsumerCount() != 1) {
                continue;
            }
            if (IsGraphOutput(graph_topo, gemm_output_edge_id)) {
                continue;
            }

            auto successor_node_id = gemm_output_edge->CreateConsumerIter().Get();
            auto successor_node = graph_topo->GetNodeById(successor_node_id);
            if (successor_node->GetType().domain != "") {
                continue;
            }

            auto gemm_kernel = reinterpret_cast<GemmOp*>(info->kernels[gemm_node->GetId()].get());
            if (successor_node->GetType().name == "Relu") {
                if (!gemm_kernel->TryFuseReLU()) { // set fuse flag to gemm_op
                    continue;
                }
            } else {
                continue;
            }

            auto activation_node = successor_node;
            auto activation_node_id = activation_node->GetId();
            auto activation_output_edge_id = activation_node->GetOutput(0);
            auto activation_output_edge = graph_topo->GetEdgeById(activation_output_edge_id);
            // gemm_node -> gemm_output_edge -> activation_node -> activation_output_edge
            // gemm_node                                      -> activation_output_edge
            gemm_node->ReplaceOutput(gemm_output_edge_id, activation_output_edge_id);
            activation_output_edge->SetProducer(gemm_node->GetId());

            // LOG(INFO) << "merge kernel " << activation_node->GetName() << " into kernel " <<
            // gemm_node->GetName() << ".";
            info->kernels.erase(activation_node_id);
            tensors.erase(gemm_output_edge_id);
            graph_topo->DelNodeById(activation_node_id);
            graph_topo->DelEdgeById(gemm_output_edge_id);

            graph_changed = true;
        }
    }

    return graph_changed;
}

}}} // namespace ppl::nn::x86

