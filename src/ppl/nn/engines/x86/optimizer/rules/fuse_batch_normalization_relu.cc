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

#include "ppl/nn/engines/x86/optimizer/rules/fuse_batch_normalization_relu.h"
#include "ppl/nn/engines/x86/optimizer/rules/utils.h"
#include "ppl/nn/engines/x86/optimizer/opt_rule_manager.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/batch_normalization_op.h"

namespace ppl { namespace nn { namespace x86 {

bool FuseBatchNormalizationReLU(const OptKernelOptions &options) {
    bool graph_changed = false;
    auto graph_topo = options.graph_topo;
    auto info = options.info;

    for (auto it = graph_topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "BatchNormalization") {
            auto bn_node = node;
            if (bn_node->GetOutputCount() > 1) { // training mode bn
                continue;
            }
            if (bn_node->GetInputCount() != 5) {
                continue;
            }
            auto bn_output_edge = graph_topo->GetEdgeById(bn_node->GetOutput(0));
            if (!bn_output_edge || bn_output_edge->CalcConsumerCount() != 1 ||
                IsGraphOutput(graph_topo, bn_output_edge->GetId())) {
                continue;
            }

            auto successor_node = graph_topo->GetNodeById(bn_output_edge->CreateConsumerIter().Get());
            if (!successor_node) {
                continue;
            }
            if (successor_node->GetType().domain != "" || successor_node->GetType().name != "Relu") {
                continue;
            }
            auto relu_node = successor_node;
            auto relu_output_edge = graph_topo->GetEdgeById(relu_node->GetOutput(0));

            auto bn_kernel_it = info->kernels.find(bn_node->GetId());
            if (bn_kernel_it == info->kernels.end()) {
                continue;
            }
            auto bn_kernel = (BatchNormalizationOp*)bn_kernel_it->second.get();

            // bn_node -> bn_output_edge -> relu_node -> relu_output_edge
            bn_kernel->TryFuseReLU();
            bn_node->ReplaceOutput(bn_output_edge->GetId(), relu_output_edge->GetId());
            relu_output_edge->SetProducer(bn_node->GetId());

            // LOG(INFO) << "merge kernel " << bn_node->GetName() << " and " << relu_node->GetName() << ".";
            info->kernels.erase(relu_node->GetId());
            graph_topo->DelNodeById(relu_node->GetId());
            graph_topo->DelEdgeById(bn_output_edge->GetId());

            graph_changed = true;
        }
    }

    return graph_changed;
}

}}} // namespace ppl::nn::x86

