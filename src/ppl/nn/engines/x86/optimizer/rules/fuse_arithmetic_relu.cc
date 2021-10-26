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
#include "ppl/nn/engines/x86/optimizer/ops/onnx/add_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/mul_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/sub_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/div_op.h"

namespace ppl { namespace nn { namespace x86 {

bool FuseArithmeticReLU(const OptKernelOptions &options) {
    bool graph_changed = false;
    auto graph_topo = options.graph_topo;
    auto info = options.info;
    auto &tensors = *options.tensors;

    for (auto it = graph_topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        enum arithmetic_type { ADD, SUB, MUL, DIV };
        if (node->GetType().domain == "" &&
            (node->GetType().name == "Add" || node->GetType().name == "Sub" || node->GetType().name == "Mul" ||
             node->GetType().name == "Div")) {
            arithmetic_type at;
            if (node->GetType().name == "Add") {
                at = ADD;
            } else if (node->GetType().name == "Sub") {
                at = SUB;
            } else if (node->GetType().name == "Mul") {
                at = MUL;
            } else if (node->GetType().name == "Div") {
                at = DIV;
            } else {
                continue;
            }

            auto arithmetic_node = node;
            auto arithmetic_output_edge = graph_topo->GetEdgeById(arithmetic_node->GetOutput(0));
            if (!arithmetic_output_edge || arithmetic_output_edge->CalcConsumerCount() != 1 ||
                IsGraphOutput(graph_topo, arithmetic_output_edge->GetId())) {
                continue;
            }
            auto arithmetic_output_edge_shape = tensors[arithmetic_output_edge->GetId()]->GetShape();

            // Only Support FP32
            if (arithmetic_output_edge_shape.GetDataType() != ppl::common::DATATYPE_FLOAT32) {
                continue;
            }

            auto successor_node = graph_topo->GetNodeById(arithmetic_output_edge->CreateConsumerIter().Get());
            if (!successor_node) {
                continue;
            }
            if (successor_node->GetType().domain != "" || successor_node->GetType().name != "Relu") {
                continue;
            }
            auto relu_node = successor_node;
            auto relu_output_edge = graph_topo->GetEdgeById(relu_node->GetOutput(0));

            auto arithmetic_kernel_it = info->kernels.find(arithmetic_node->GetId());
            if (arithmetic_kernel_it == info->kernels.end()) {
                continue;
            }

            // arithmetic_node -> arithmetic_output_edge -> relu_node -> relu_output_edge
            if (at == ADD) {
                auto arithmetic_kernel = (AddOp*)arithmetic_kernel_it->second.get();
                arithmetic_kernel->TryFuseReLU();
            } else if (at == SUB) {
                auto arithmetic_kernel = (SubOp*)arithmetic_kernel_it->second.get();
                arithmetic_kernel->TryFuseReLU();
            } else if (at == MUL) {
                auto arithmetic_kernel = (MulOp*)arithmetic_kernel_it->second.get();
                arithmetic_kernel->TryFuseReLU();
            } else if (at == DIV) {
                auto arithmetic_kernel = (DivOp*)arithmetic_kernel_it->second.get();
                arithmetic_kernel->TryFuseReLU();
            } else {
                continue;
            }

            arithmetic_node->ReplaceOutput(arithmetic_output_edge->GetId(), relu_output_edge->GetId());
            relu_output_edge->SetProducer(arithmetic_node->GetId());

            info->kernels.erase(relu_node->GetId());
            graph_topo->DelNodeById(relu_node->GetId());
            graph_topo->DelEdgeById(arithmetic_output_edge->GetId());

            graph_changed = true;
        }
    }

    return graph_changed;
}

}}} // namespace ppl::nn::x86

