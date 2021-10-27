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

#include "ppl/nn/engines/x86/optimizer/rules/fuse_channel_shuffle.h"
#include "ppl/nn/engines/x86/optimizer/rules/utils.h"
#include "ppl/nn/engines/x86/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/x86/optimizer/opt_rule_manager.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/x86/optimizer/ops/ppl/channel_shuffle_op.h"
#include "ppl/nn/params/onnx/transpose_param.h"
#include "ppl/nn/params/onnx/concat_param.h"
#include "ppl/nn/params/onnx/split_param.h"

namespace ppl { namespace nn { namespace x86 {

bool FuseChannelShuffle(const OptKernelOptions &options) {
    bool graph_changed = false;
    auto graph_topo = options.graph_topo;
    auto graph_data = options.graph_data;
    auto info = options.info;
    auto &tensors = *options.tensors;

    for (auto it = graph_topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "Reshape") {
            // find 1st Reshape node
            auto reshape1_node = node;
            auto reshape1_node_id = reshape1_node->GetId();
            auto reshape1_output_edge_id = reshape1_node->GetOutput(0);
            auto reshape1_output_edge = graph_topo->GetEdgeById(reshape1_output_edge_id);

            if (reshape1_output_edge->CalcConsumerCount() != 1) {
                continue;
            }
            if (IsGraphOutput(graph_topo, reshape1_output_edge_id)) {
                continue;
            }

            // find transpose node
            auto successor_node_id = reshape1_output_edge->CreateConsumerIter().Get();
            auto successor_node = graph_topo->GetNodeById(successor_node_id);
            if (successor_node == nullptr || successor_node->GetType().domain != "" || successor_node->GetType().name != "Transpose") {
                continue;
            }
            auto trans_node_id = successor_node_id;
            auto trans_node = successor_node;
            auto trans_output_edge_id = trans_node->GetOutput(0);
            auto trans_output_edge = graph_topo->GetEdgeById(trans_output_edge_id);
            if (trans_output_edge->CalcConsumerCount() != 1) {
                continue;
            }
            if (IsGraphOutput(graph_topo, trans_output_edge_id)) {
                continue;
            }

            // find 2nd reshape node
            successor_node_id = trans_output_edge->CreateConsumerIter().Get();
            successor_node = graph_topo->GetNodeById(successor_node_id);
            if (successor_node == nullptr || successor_node->GetType().domain != "" || successor_node->GetType().name != "Reshape") {
                continue;
            }
            auto reshape2_node = successor_node;
            auto reshape2_node_id = reshape2_node->GetId();
            auto reshape2_output_edge_id = reshape2_node->GetOutput(0);
            auto reshape2_output_edge = graph_topo->GetEdgeById(reshape2_output_edge_id);
            if (IsGraphOutput(graph_topo, reshape2_output_edge_id)) {
                continue;
            }

            // check reshape input[1] kind
            auto shape1_edge_id = reshape1_node->GetInput(1);
            auto shape2_edge_id = reshape2_node->GetInput(1);
            auto shape1_edge = graph_topo->GetEdgeById(shape1_edge_id);
            auto shape2_edge = graph_topo->GetEdgeById(shape2_edge_id);
            if (graph_data->constants.find(shape1_edge_id) == graph_data->constants.end() ||
                graph_data->constants.find(shape2_edge_id) == graph_data->constants.end()) {
                continue;
            }

            // reshape size check
            auto& reshape1_output_shape = tensors[reshape1_output_edge_id]->GetShape();
            auto& reshape2_output_shape = tensors[reshape2_output_edge_id]->GetShape();

            if (reshape1_output_shape.IsEmpty() ||
                reshape2_output_shape.IsEmpty()) { // input shape has not been infered
                continue;
            }

            if (reshape1_output_shape.GetDimCount() != 5 ||
                reshape1_output_shape.GetDimCount() - reshape2_output_shape.GetDimCount() != 1) {
                continue;
            }

            if (reshape1_output_shape.GetDim(1) * reshape1_output_shape.GetDim(2) != reshape2_output_shape.GetDim(1) ||
                reshape1_output_shape.GetDim(0) != reshape2_output_shape.GetDim(0)) {
                continue;
            }

            if (reshape1_output_shape.GetDim(3) != reshape2_output_shape.GetDim(2) ||
                reshape1_output_shape.GetDim(4) != reshape2_output_shape.GetDim(3)) {
                continue;
            }

            // check transpose attribute
            auto& attrs = graph_data->attrs;
            if (attrs.find(trans_node_id) == attrs.end()) {
                continue;
            }
            common::TransposeParam* transpose_param = (common::TransposeParam*)attrs[trans_node_id].get();
            auto perm = transpose_param->perm;
            if (perm.size() != 5) {
                continue;
            }
            if (perm[0] != 0 || perm[1] != 2 || perm[2] != 1 || perm[3] != 3 || perm[4] != 4) {
                continue;
            }

            bool fuse_concat = false;
            bool fuse_split = false;
            bool fuse_concat_split = false;
            auto reshape1_input_edge = graph_topo->GetEdgeById(reshape1_node->GetInput(0));
            auto reshape1_prev_node = graph_topo->GetNodeById(reshape1_input_edge->GetProducer());
            auto reshape2_next_node = graph_topo->GetNodeById(reshape2_output_edge->CreateConsumerIter().Get());
            auto concat_param = (common::ConcatParam*)attrs[reshape1_prev_node->GetId()].get();
            auto split_param = (common::SplitParam*)attrs[reshape2_next_node->GetId()].get();
            if (true
                && reshape1_input_edge->CalcConsumerCount() == 1
                && reshape1_prev_node != nullptr
                && reshape1_prev_node->GetType().domain == ""
                && reshape1_prev_node->GetType().name == "Concat"
                && reshape1_prev_node->GetInputCount() == 2
                && concat_param->axis == 1)
            {
                fuse_concat = true;
            }
            if (true
                && reshape2_output_edge->CalcConsumerCount() == 1
                && reshape2_next_node != nullptr
                && reshape2_next_node->GetType().domain == ""
                && reshape2_next_node->GetType().name == "Split"
                && reshape2_next_node->GetOutputCount() == 2
                && split_param->axis == 1
                && split_param->split_point[0] == split_param->split_point[1])
            {
                fuse_split = true;
            }
            fuse_concat_split = fuse_concat && fuse_split;

            std::string channel_shuffle_node_name =
                "ChannelShuffle_" +
                (fuse_concat ? (reshape1_prev_node->GetName() + "_") : "") +
                reshape1_node->GetName() + "_" +
                trans_node->GetName() + "_" +
                reshape2_node->GetName() +
                (fuse_split ? ("_" + reshape2_next_node->GetName()) : "");
            auto node_ret_pair = graph_topo->AddNode(channel_shuffle_node_name);
            if (!node_ret_pair.second) {
                LOG(ERROR) << "node[" << channel_shuffle_node_name << "] already exists.";
                continue;
            }
            ir::Node* channel_shuffle_node = node_ret_pair.first;
            channel_shuffle_node->SetType(ir::Node::Type("ppl", "ChannelShuffle", 1));

            auto param_ref = options.graph_data->attrs.find(channel_shuffle_node->GetId());
            if (param_ref == options.graph_data->attrs.end()) {
                auto channel_shuffle_param = std::make_shared<ppl::nn::common::ChannelShuffleParam>();
                const int32_t group = (int32_t)reshape1_output_shape.GetDim(1);
                channel_shuffle_param->group = group;
                options.graph_data->attrs[channel_shuffle_node->GetId()] = channel_shuffle_param;
            } else {
                LOG(ERROR) << "Node " << channel_shuffle_node->GetName() << "param exist.";
                continue;
            }

            if (fuse_concat_split) {
                // add ChannelShuffle node into graph
                // base_edge1/2 -> concat_node -> concat_edge -> reshape1_node -> reshape1_edge -> trans_node -> trans_edge -> reshape2_node -> reshape2_edge -> split_node -> split_edge1/2

                // base_edge1/2 -> ChannelShufflenode                                                                                                                       -> split_edge1/2
                auto base_edge1_id = reshape1_prev_node->GetInput(0);
                auto base_edge1 = graph_topo->GetEdgeById(base_edge1_id);
                auto base_edge2_id = reshape1_prev_node->GetInput(1);
                auto base_edge2 = graph_topo->GetEdgeById(base_edge2_id);

                auto split_edge1_id = reshape2_next_node->GetOutput(0);
                auto split_edge1 = graph_topo->GetEdgeById(split_edge1_id);
                auto split_edge2_id = reshape2_next_node->GetOutput(1);
                auto split_edge2 = graph_topo->GetEdgeById(split_edge2_id);

                channel_shuffle_node->AddInput(base_edge1_id);
                channel_shuffle_node->AddInput(base_edge2_id);
                channel_shuffle_node->AddOutput(split_edge1_id);
                channel_shuffle_node->AddOutput(split_edge2_id);

                X86OptKernel *opt_kernel = nullptr;
                if (ppl::common::RC_SUCCESS != CreateX86OptKernel(options, channel_shuffle_node, &opt_kernel)) {
                    LOG(ERROR) << "Node " << channel_shuffle_node->GetName() << "param exist.";
                    graph_topo->DelNodeById(channel_shuffle_node->GetId());
                    continue;
                }

                base_edge1->DelConsumer(reshape1_prev_node->GetId());
                base_edge1->AddConsumer(channel_shuffle_node->GetId());
                base_edge2->DelConsumer(reshape1_prev_node->GetId());
                base_edge2->AddConsumer(channel_shuffle_node->GetId());

                split_edge1->SetProducer(channel_shuffle_node->GetId());
                split_edge2->SetProducer(channel_shuffle_node->GetId());

                info->kernels.erase(reshape1_prev_node->GetId());
                info->kernels.erase(reshape1_node_id);
                info->kernels.erase(trans_node_id);
                info->kernels.erase(reshape2_node_id);
                info->kernels.erase(reshape2_next_node->GetId());

                graph_topo->DelEdgeById(reshape1_input_edge->GetId());
                graph_topo->DelEdgeById(reshape1_output_edge_id);
                graph_topo->DelEdgeById(trans_output_edge_id);
                graph_topo->DelEdgeById(reshape2_output_edge_id);

                graph_topo->DelNodeById(reshape1_prev_node->GetId());
                graph_topo->DelNodeById(reshape1_node_id);
                graph_topo->DelNodeById(trans_node_id);
                graph_topo->DelNodeById(reshape2_node_id);
                graph_topo->DelNodeById(reshape2_next_node->GetId());
            } else if (fuse_concat) {
                // add ChannelShuffle node into graph
                // base_edge1/2 -> concat_node -> concat_edge -> reshape1_node -> reshape1_edge -> trans_node -> trans_edge -> reshape2_node -> reshape2_edge

                // base_edge1/2 -> ChannelShufflenode                                                                                        -> reshape2_edge
                auto base_edge1_id = reshape1_prev_node->GetInput(0);
                auto base_edge1 = graph_topo->GetEdgeById(base_edge1_id);
                auto base_edge2_id = reshape1_prev_node->GetInput(1);
                auto base_edge2 = graph_topo->GetEdgeById(base_edge2_id);

                channel_shuffle_node->AddInput(base_edge1_id);
                channel_shuffle_node->AddInput(base_edge2_id);
                channel_shuffle_node->AddOutput(reshape2_output_edge_id);

                X86OptKernel *opt_kernel = nullptr;
                if (ppl::common::RC_SUCCESS != CreateX86OptKernel(options, channel_shuffle_node, &opt_kernel)) {
                    LOG(ERROR) << "Node " << channel_shuffle_node->GetName() << "param exist.";
                    graph_topo->DelNodeById(channel_shuffle_node->GetId());
                    continue;
                }

                base_edge1->DelConsumer(reshape1_prev_node->GetId());
                base_edge1->AddConsumer(channel_shuffle_node->GetId());
                base_edge2->DelConsumer(reshape1_prev_node->GetId());
                base_edge2->AddConsumer(channel_shuffle_node->GetId());

                reshape2_output_edge->SetProducer(channel_shuffle_node->GetId());

                info->kernels.erase(reshape1_prev_node->GetId());
                info->kernels.erase(reshape1_node_id);
                info->kernels.erase(trans_node_id);
                info->kernels.erase(reshape2_node_id);

                graph_topo->DelEdgeById(reshape1_input_edge->GetId());
                graph_topo->DelEdgeById(reshape1_output_edge_id);
                graph_topo->DelEdgeById(trans_output_edge_id);

                graph_topo->DelNodeById(reshape1_prev_node->GetId());
                graph_topo->DelNodeById(reshape1_node_id);
                graph_topo->DelNodeById(trans_node_id);
                graph_topo->DelNodeById(reshape2_node_id);
            } else {
                // add ChannelShuffle node into graph
                // base_node -> base_edge -> reshape1_node -> reshape1_edge -> trans_node -> trans_edge -> reshape2_node -> reshape2_edge

                // base_node -> base_edge -> ChannelShufflenode                                                          -> reshape2_edge
                auto base_edge_id = reshape1_node->GetInput(0);
                auto base_edge = graph_topo->GetEdgeById(base_edge_id);

                channel_shuffle_node->AddInput(base_edge_id);
                channel_shuffle_node->AddOutput(reshape2_output_edge_id);

                X86OptKernel *opt_kernel = nullptr;
                if (ppl::common::RC_SUCCESS != CreateX86OptKernel(options, channel_shuffle_node, &opt_kernel)) {
                    LOG(ERROR) << "Node " << channel_shuffle_node->GetName() << "param exist.";
                    graph_topo->DelNodeById(channel_shuffle_node->GetId());
                    continue;
                }

                base_edge->DelConsumer(reshape1_node_id);
                base_edge->AddConsumer(channel_shuffle_node->GetId());

                reshape2_output_edge->SetProducer(channel_shuffle_node->GetId());

                info->kernels.erase(reshape1_node_id);
                info->kernels.erase(trans_node_id);
                info->kernels.erase(reshape2_node_id);

                graph_topo->DelEdgeById(reshape1_output_edge_id);
                graph_topo->DelEdgeById(trans_output_edge_id);

                graph_topo->DelNodeById(reshape1_node_id);
                graph_topo->DelNodeById(trans_node_id);
                graph_topo->DelNodeById(reshape2_node_id);
            }

            shape1_edge->DelConsumer(reshape1_node_id);
            shape2_edge->DelConsumer(reshape2_node_id);

            if (shape1_edge->CalcConsumerCount() == 0 && !IsGraphOutput(graph_topo, shape1_edge->GetId())) {
                graph_data->constants.erase(shape1_edge_id);
                graph_topo->DelEdgeById(shape1_edge_id);
            }
            if (shape2_edge->CalcConsumerCount() == 0 && !IsGraphOutput(graph_topo, shape2_edge->GetId())) {
                graph_data->constants.erase(shape2_edge_id);
                graph_topo->DelEdgeById(shape2_edge_id);
            }

            graph_changed = true;
        }
    }

    return graph_changed;
}

}}} // namespace ppl::nn::x86

