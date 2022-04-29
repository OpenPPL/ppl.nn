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

#include "ppl/nn/engines/x86/optimizer/rules/fuse_conv_depthwise.h"
#include "ppl/nn/engines/x86/optimizer/rules/utils.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/conv_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/pmx/post_depthwise_conv_op.h"

namespace ppl { namespace nn { namespace x86 {

bool FuseConvDepthwise(const OptKernelOptions &options) {
    bool graph_changed = false;
    auto graph_topo = options.graph_topo;
    auto info = options.info;
    auto &tensors = *options.tensors;

    for (auto it = graph_topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "Conv") {
            auto conv_node = node;
            auto conv_output_edge_id = conv_node->GetOutput(0);
            auto conv_out_edge = graph_topo->GetEdge(conv_output_edge_id);
            if (conv_out_edge->CalcConsumerCount() != 1) {
                continue;
            }
            if (IsReservedEdge(tensors, conv_output_edge_id)) {
                continue;
            }
            auto next_node = graph_topo->GetNode(conv_out_edge->CreateConsumerIter().Get());
            if (next_node->GetType().domain != "" || next_node->GetType().name != "Conv") {
                continue;
            }

            auto conv_kernel = reinterpret_cast<ConvOp*>(info->kernels[conv_node->GetId()].get());
            auto post_conv_kernel = reinterpret_cast<ConvOp*>(info->kernels[next_node->GetId()].get());
            auto pd_conv2d_param = PostDepthwiseConvOp::TryMakePostDepthwiseConv2dParam(conv_kernel, post_conv_kernel);

            if (pd_conv2d_param == nullptr) {
                continue;
            }

            const std::string pd_conv2d_node_name =
                    "PostDepthwiseConv_" + conv_node->GetName() + "_" + next_node->GetName();
            const ir::Node::Type type("pmx", "PostDepthwiseConv", 1);

            // add node to graph topo
            auto node_ret_pair = graph_topo->AddNode(pd_conv2d_node_name);
            if (!node_ret_pair.second) {
                LOG(ERROR) << "node[" << pd_conv2d_node_name << "] already exists.";
                continue;
            }
            auto pd_conv2d_node = node_ret_pair.first;
            pd_conv2d_node->SetType(type);

            // add new node input/output
            bool conv_has_bias = conv_kernel->GetBiasTerm();
            bool depthwise_has_bias = post_conv_kernel->GetBiasTerm();

            auto conv_input = graph_topo->GetEdge(conv_node->GetInput(0));
            auto conv_w = graph_topo->GetEdge(conv_node->GetInput(1));
            auto conv_b = conv_has_bias ? graph_topo->GetEdge(conv_node->GetInput(2)) : nullptr;
            auto conv_output = graph_topo->GetEdge(conv_node->GetOutput(0)); // depthwise input
            auto depthwise_w = graph_topo->GetEdge(next_node->GetInput(1));
            auto depthwise_b = depthwise_has_bias ? graph_topo->GetEdge(next_node->GetInput(2)) : nullptr;
            auto depthwise_output = graph_topo->GetEdge(next_node->GetOutput(0));
            pd_conv2d_node->AddInput(conv_input->GetId());
            pd_conv2d_node->AddOutput(depthwise_output->GetId());

            // create opt kernel & set param and dataformat
            X86OptKernel *opt_kernel = nullptr;
            auto status = CreateX86OptKernel(options, pd_conv2d_node, &opt_kernel);
            if (status != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "Create OptKernel [" << pd_conv2d_node_name << "] failed: " << ppl::common::GetRetCodeStr(status);
                graph_topo->DelNode(node->GetId());
                continue;
            }

            auto pd_conv2d_kernel = reinterpret_cast<PostDepthwiseConvOp*>(opt_kernel);
            pd_conv2d_kernel->SetPostDepthwiseConv2dParam(pd_conv2d_param);
            pd_conv2d_kernel->SetOutputDataFormat( // save data format to pd_conv2d
                    0, tensors[next_node->GetOutput(0)].get()->GetShape()->GetDataFormat());

            // change graph topo
            conv_input->DelConsumer(conv_node->GetId());
            conv_w->DelConsumer(conv_node->GetId());
            depthwise_w->DelConsumer(next_node->GetId());
            if (conv_has_bias) conv_b->DelConsumer(conv_node->GetId());
            if (depthwise_has_bias) depthwise_b->DelConsumer(next_node->GetId());
            // ====
            depthwise_output->SetProducer(pd_conv2d_node->GetId());
            // ====
            conv_input->AddConsumer(pd_conv2d_node->GetId());

            // delete kernel & tensors
            bool del_conv_w = conv_w->CalcConsumerCount() == 0;
            bool del_conv_b = conv_has_bias && conv_b->CalcConsumerCount() == 0;
            bool del_depthwise_w = depthwise_w->CalcConsumerCount() == 0;
            bool del_depthwise_b = depthwise_has_bias && depthwise_b->CalcConsumerCount() == 0;
            info->kernels.erase(conv_node->GetId());
            info->kernels.erase(next_node->GetId());
            tensors.erase(conv_output->GetId());
            if (del_conv_w) tensors.erase(conv_w->GetId());
            if (del_conv_b) tensors.erase(conv_b->GetId());
            if (del_depthwise_w) tensors.erase(depthwise_w->GetId());
            if (del_depthwise_b) tensors.erase(depthwise_b->GetId());

            // delete unused node & edge
            graph_topo->DelNode(conv_node->GetId());
            graph_topo->DelNode(next_node->GetId());
            graph_topo->DelEdge(conv_output->GetId());
            auto conv_w_id = conv_w->GetId();
            auto conv_b_id = conv_b->GetId();
            auto depthwise_w_id = depthwise_w->GetId();
            auto depthwise_b_id = depthwise_b->GetId();
            if (del_conv_w) graph_topo->DelEdge(conv_w_id);
            if (del_conv_b) graph_topo->DelEdge(conv_b_id);
            if (del_depthwise_w) graph_topo->DelEdge(depthwise_w_id);
            if (del_depthwise_b) graph_topo->DelEdge(depthwise_b_id);

            graph_changed = true;
        }
    }

    return graph_changed;
}

}}} // namespace ppl::nn::x86

