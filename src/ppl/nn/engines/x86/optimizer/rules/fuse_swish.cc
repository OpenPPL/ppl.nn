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

#include "ppl/nn/engines/x86/optimizer/rules/fuse_swish.h"
#include "ppl/nn/engines/x86/optimizer/rules/utils.h"
#include "ppl/nn/engines/x86/optimizer/opt_rule_manager.h"
#include "ppl/nn/engines/x86/optimizer/ops/ppl/swish_op.h"
#include "ppl/nn/common/logger.h"

namespace ppl { namespace nn { namespace x86 {

bool FuseSwish(const OptKernelOptions &options) {
    bool graph_changed = false;
    auto graph_topo = options.graph_topo;
    auto info = options.info;
    auto &tensors = *options.tensors;

    for (auto it = graph_topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "Sigmoid") {
            auto sigmoid_node = node;
            auto sigmoid_output_edge = graph_topo->GetEdgeById(sigmoid_node->GetOutput(0));
            if (sigmoid_output_edge->CalcConsumerCount() != 1 || IsGraphOutput(graph_topo, sigmoid_output_edge->GetId())) {
                continue;
            }
            auto sigmoid_input_edge = graph_topo->GetEdgeById(sigmoid_node->GetInput(0));

            auto successor_node = graph_topo->GetNodeById(sigmoid_output_edge->CreateConsumerIter().Get());
            if (!successor_node || successor_node->GetType().domain != "" || successor_node->GetType().name != "Mul") {
                continue;
            }
            auto last_mul_node = successor_node;

            auto last_mul_input_edge0 = graph_topo->GetEdgeById(last_mul_node->GetInput(0));
            auto last_mul_input_edge1 = graph_topo->GetEdgeById(last_mul_node->GetInput(1));
            if (!last_mul_input_edge0 || !last_mul_input_edge1) {
                continue;
            }
            auto last_mul_output_edge = graph_topo->GetEdgeById(last_mul_node->GetOutput(0));

            ir::Edge* last_mul_inner_input_edge;
            ir::Edge* last_mul_outer_input_edge;
            if (last_mul_input_edge0 == sigmoid_output_edge) {
                last_mul_inner_input_edge = last_mul_input_edge0;
                last_mul_outer_input_edge = last_mul_input_edge1;
            } else {
                last_mul_inner_input_edge = last_mul_input_edge1;
                last_mul_outer_input_edge = last_mul_input_edge0;
            }
            if (last_mul_inner_input_edge != sigmoid_output_edge) {
                continue;
            }

            if (last_mul_outer_input_edge == sigmoid_input_edge) {
                // swish without beta (beta default to 1)
                // its pattern is:
                // sigmoid_input_edge(last_mul_outer_input_edge) ----> sigmoid_node ----> sigmoid_output_edge(last_mul_inner_input_edge) ----> last_mul_node ----> last_mul_output_edge
                //                                               |-------------------------------------------------------------------------->|
                const std::string swish_node_name =
                    "Fused_Swish_" + sigmoid_node->GetName() + "_" + last_mul_node->GetName();
                const ir::Node::Type type("ppl", "Swish", 1);

                // add node to graph topo
                auto node_ret_pair = graph_topo->AddNode(swish_node_name);
                if (!node_ret_pair.second) {
                    LOG(ERROR) << "node[" << swish_node_name << "] already exists.";
                    continue;
                }
                auto swish_node = node_ret_pair.first;
                swish_node->SetType(type);

                // add new node input/output
                swish_node->AddInput(sigmoid_input_edge->GetId());
                swish_node->AddOutput(last_mul_output_edge->GetId());

                // create opt kernel & set param and dataformat
                X86OptKernel* swish_opt_kernel = nullptr;
                auto status = CreateX86OptKernel(options, swish_node, &swish_opt_kernel);
                if (status != ppl::common::RC_SUCCESS) {
                    LOG(ERROR) << "Create OptKernel [" << swish_node_name << "] failed: " << ppl::common::GetRetCodeStr(status);
                    graph_topo->DelNodeById(node->GetId());
                    continue;
                }

                ((SwishOp*)swish_opt_kernel)->SetBeta(1.0f); // default to 1.0
                swish_opt_kernel->SetOutputDataFormat(
                    0, tensors[last_mul_output_edge->GetId()].get()->GetShape().GetDataFormat());

                // change graph topo
                sigmoid_input_edge->DelConsumer(sigmoid_node->GetId());
                sigmoid_input_edge->DelConsumer(last_mul_node->GetId());
                sigmoid_input_edge->AddConsumer(swish_node->GetId());
                last_mul_output_edge->SetProducer(swish_node->GetId());

                // delete unused node & edge
                info->kernels.erase(sigmoid_node->GetId());
                info->kernels.erase(last_mul_node->GetId());
                tensors.erase(sigmoid_output_edge->GetId());

                // LOG(INFO) << "successfully merged node " << sigmoid_node->GetName() << " and "
                //            << last_mul_node->GetName() << " into node " << swish_node->GetName() << ".";
                graph_topo->DelEdgeById(sigmoid_output_edge->GetId());
                graph_topo->DelNodeById(sigmoid_node->GetId());
                graph_topo->DelNodeById(last_mul_node->GetId());

                graph_changed = true;
            }

            // TODO: fuse swish with beta(another mul op)
        }
    }

    return graph_changed;
}

}}} // namespace ppl::nn::x86

