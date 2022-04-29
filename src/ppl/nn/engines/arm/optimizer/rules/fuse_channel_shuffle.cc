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

#include "ppl/nn/engines/arm/optimizer/rules/fuse_channel_shuffle.h"
#include "ppl/nn/engines/arm/optimizer/rules/utils.h"
#include "ppl/nn/params/onnx/transpose_param.h"
#include "ppl/nn/params/onnx/concat_param.h"
#include "ppl/nn/params/onnx/split_param.h"
#include "ppl/nn/params/pmx/channel_shuffle_param.h"

namespace ppl { namespace nn { namespace arm {

inline void GetSliceParam(const ir::Node* node, const ir::GraphTopo* graph_topo, const ir::GraphData* graph_data,
                          std::vector<int64_t>& starts, std::vector<int64_t>& ends, std::vector<int64_t>& axes,
                          std::vector<int64_t>& steps) {
    if (node == nullptr) {
        return;
    }
    if (node->GetType().domain != "" || node->GetType().name != "Slice") {
        return;
    }

    const auto& constants = graph_data->constants;

    auto starts_edge = graph_topo->GetEdge(node->GetInput(1));
    if (starts_edge != nullptr && constants.find(starts_edge->GetId()) != constants.end()) {
        auto it = constants.find(starts_edge->GetId());
        const auto starts_data = it->second.data;
        starts = std::vector<int64_t>((const int64_t*)starts_data.data(),
                                      (const int64_t*)starts_data.data() + starts_data.size() / sizeof(int64_t));
    }

    auto ends_edge = graph_topo->GetEdge(node->GetInput(2));
    if (ends_edge != nullptr && constants.find(ends_edge->GetId()) != constants.end()) {
        auto it = constants.find(ends_edge->GetId());
        const auto ends_data = it->second.data;
        ends = std::vector<int64_t>((const int64_t*)ends_data.data(),
                                    (const int64_t*)ends_data.data() + ends_data.size() / sizeof(int64_t));
    }

    auto axes_edge = node->GetInputCount() >= 4 ? graph_topo->GetEdge(node->GetInput(3)) : nullptr;
    if (axes_edge != nullptr && constants.find(axes_edge->GetId()) != constants.end()) {
        auto it = constants.find(axes_edge->GetId());
        const auto axes_data = it->second.data;
        axes = std::vector<int64_t>((const int64_t*)axes_data.data(),
                                    (const int64_t*)axes_data.data() + axes_data.size() / sizeof(int64_t));
    }

    auto steps_edge = node->GetInputCount() >= 5 ? graph_topo->GetEdge(node->GetInput(4)) : nullptr;
    if (steps_edge != nullptr && constants.find(steps_edge->GetId()) != constants.end()) {
        auto it = constants.find(steps_edge->GetId());
        const auto steps_data = it->second.data;
        steps = std::vector<int64_t>((const int64_t*)steps_data.data(),
                                     (const int64_t*)steps_data.data() + steps_data.size() / sizeof(int64_t));
    }
}

bool FuseChannelShuffleRule::Apply(const OptKernelOptions& options) {
    bool graph_changed = false;

    auto graph_topo = options.graph_topo;
    auto graph_data = options.graph_data;
    auto& tensors = *options.tensors;

    for (auto it = graph_topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "Reshape") { // start from reshape op
            /******************** pattern match ***********************/
            /** 1. check topo **/
            // find 1st Reshape node
            auto reshape1_node = node;
            auto reshape1_output_edge_id = reshape1_node->GetOutput(0);
            auto reshape1_output_edge = graph_topo->GetEdge(reshape1_output_edge_id);

            if (reshape1_output_edge->CalcConsumerCount() != 1) {
                continue;
            }
            if (IsGraphInput(graph_topo, reshape1_output_edge_id) ||
                IsGraphOutput(graph_topo, reshape1_output_edge_id)) {
                continue;
            }

            // find transpose node
            auto successor_node_id = reshape1_output_edge->CreateConsumerIter().Get();
            auto successor_node = graph_topo->GetNode(successor_node_id);
            if (successor_node == nullptr || successor_node->GetType().domain != "" ||
                successor_node->GetType().name != "Transpose") {
                continue;
            }
            auto trans_node_id = successor_node_id;
            auto trans_node = successor_node;
            auto trans_output_edge_id = trans_node->GetOutput(0);
            auto trans_output_edge = graph_topo->GetEdge(trans_output_edge_id);
            if (trans_output_edge->CalcConsumerCount() != 1) {
                continue;
            }
            if (IsGraphInput(graph_topo, trans_output_edge_id) || IsGraphOutput(graph_topo, trans_output_edge_id)) {
                continue;
            }

            // find 2nd reshape node
            successor_node_id = trans_output_edge->CreateConsumerIter().Get();
            successor_node = graph_topo->GetNode(successor_node_id);
            if (successor_node == nullptr || successor_node->GetType().domain != "" ||
                successor_node->GetType().name != "Reshape") {
                continue;
            }
            auto reshape2_node = successor_node;
            auto reshape2_output_edge_id = reshape2_node->GetOutput(0);
            auto reshape2_output_edge = graph_topo->GetEdge(reshape2_output_edge_id);
            if (IsGraphInput(graph_topo, reshape2_output_edge_id) ||
                IsGraphOutput(graph_topo, reshape2_output_edge_id)) {
                continue;
            }

            /** 2. check parameter **/
            // check reshape input[1] kind
            auto shape1_edge_id = reshape1_node->GetInput(1);
            auto shape2_edge_id = reshape2_node->GetInput(1);
            if (graph_data->constants.find(shape1_edge_id) == graph_data->constants.end() ||
                graph_data->constants.find(shape2_edge_id) == graph_data->constants.end()) {
                continue;
            }

            // reshape size check
            auto& reshape1_output_shape = *tensors[reshape1_output_edge_id]->GetShape();
            auto& reshape2_output_shape = *tensors[reshape2_output_edge_id]->GetShape();

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
            auto transpose_param = (ppl::nn::onnx::TransposeParam*)attrs[trans_node_id].get();
            auto perm = transpose_param->perm;
            if (perm.size() != 5) {
                continue;
            }
            if (perm[0] != 0 || perm[1] != 2 || perm[2] != 1 || perm[3] != 3 || perm[4] != 4) {
                continue;
            }

            const int32_t group = (int32_t)reshape1_output_shape.GetDim(1);
            const int32_t channels_per_group = (int32_t)reshape1_output_shape.GetDim(2);
            const int32_t channels = group * channels_per_group;

            /** 3. check prev concat & next split/slice **/
            bool fuse_concat = false;
            bool fuse_split = false;
            bool fuse_slice = false;

            // check prev concat
            auto reshape1_input_edge = graph_topo->GetEdge(reshape1_node->GetInput(0));
            auto reshape1_prev_node = graph_topo->GetNode(reshape1_input_edge->GetProducer());
            if (reshape1_input_edge->CalcConsumerCount() == 1 && reshape1_prev_node != nullptr &&
                reshape1_prev_node->GetType().domain == "" && reshape1_prev_node->GetType().name == "Concat" &&
                reshape1_prev_node->GetInputCount() == 2) // only support two input concat
            {
                auto concat_param = (ppl::nn::onnx::ConcatParam*)attrs[reshape1_prev_node->GetId()].get();
                if (concat_param->axis == 1) {
                    int32_t sum_channels = 0;
                    for (uint32_t i = 0; i < reshape1_prev_node->GetInputCount(); i++) {
                        const TensorShape& input_shape = *tensors[reshape1_prev_node->GetInput(i)]->GetShape();
                        if (!input_shape.IsEmpty()) {
                            sum_channels += input_shape.GetDim(1);
                        }
                    }
                    if (sum_channels == channels) {
                        fuse_concat = true;
                    }
                }
            }

            // check next split/slice
            std::vector<ir::Node*> reshape2_next_nodes;
            for (auto consumer_it = reshape2_output_edge->CreateConsumerIter(); consumer_it.IsValid();
                 consumer_it.Forward()) {
                reshape2_next_nodes.push_back(graph_topo->GetNode(consumer_it.Get()));
            }

            if (reshape2_next_nodes.size() == 1) { // check next split
                auto reshape2_next_node = reshape2_next_nodes[0];
                if (reshape2_next_node != nullptr && reshape2_next_node->GetType().domain == "" &&
                    reshape2_next_node->GetType().name == "Split" &&
                    reshape2_next_node->GetOutputCount() == (uint32_t)group) // only support 2 output split
                {
                    auto split_param = (ppl::nn::onnx::SplitParam*)attrs[reshape2_next_node->GetId()].get();
                    if (split_param->axis == 1) {
                        int32_t sum_channels = 0;
                        for (uint32_t i = 0; i < reshape2_next_node->GetOutputCount(); i++) {
                            const TensorShape& output_shape = *tensors[reshape2_next_node->GetOutput(i)]->GetShape();
                            if (!output_shape.IsEmpty() || output_shape.GetDim(1) != channels_per_group) {
                                sum_channels += output_shape.GetDim(1);
                            }
                        }
                        if (sum_channels == channels) {
                            fuse_split = true;
                        }
                    }
                }
            } else if (reshape2_next_nodes.size() == 2 && group == 2) { // check next two slice
                if (reshape2_next_nodes[0] != nullptr && reshape2_next_nodes[1] != nullptr &&
                    reshape2_next_nodes[0]->GetType().domain == "" && reshape2_next_nodes[1]->GetType().domain == "" &&
                    reshape2_next_nodes[0]->GetType().name == "Slice" &&
                    reshape2_next_nodes[1]->GetType().name == "Slice") {
                    std::vector<int64_t> starts0, ends0, axes0, steps0;
                    std::vector<int64_t> starts1, ends1, axes1, steps1;
                    GetSliceParam(reshape2_next_nodes[0], graph_topo, graph_data, starts0, ends0, axes0, steps0);
                    GetSliceParam(reshape2_next_nodes[1], graph_topo, graph_data, starts1, ends1, axes1, steps1);
                    if (starts0.size() == 1 && ends0.size() == 1 && axes0.size() == 1 && starts1.size() == 1 &&
                        ends1.size() == 1 && axes1.size() == 1 && ends0[0] - starts0[0] == channels_per_group &&
                        ends1[0] - starts1[0] == channels_per_group &&
                        (ends0[0] == starts1[0] || ends1[0] == starts0[0]) && axes0[0] == 1 && axes1[0] == 1 &&
                        (steps0.size() == 0 || (steps0.size() == 1 && steps0[0] == 1)) &&
                        (steps1.size() == 0 || (steps1.size() == 1 && steps1[0] == 1))) {
                        if (ends1[0] == starts0[0]) {
                            std::swap(reshape2_next_nodes[0], reshape2_next_nodes[1]);
                        }
                        fuse_slice = true;
                    }
                }
            }

            if (!fuse_concat) { // TODO: implement not fuse_concat
                continue;
            }

            /******************** do optimize ***********************/
            /** 1. create & register fused op **/
            std::string channel_shuffle_node_name = "ChannelShuffle_" +
                (fuse_concat ? (reshape1_prev_node->GetName() + "_") : "") + reshape1_node->GetName() + "_" +
                trans_node->GetName() + "_" + reshape2_node->GetName() +
                (fuse_split ? ("_" + reshape2_next_nodes[0]->GetName()) : "") +
                (fuse_slice ? ("_" + reshape2_next_nodes[0]->GetName() + "_" + reshape2_next_nodes[1]->GetName()) : "");
            auto node_ret_pair = graph_topo->AddNode(channel_shuffle_node_name);
            if (!node_ret_pair.second) {
                LOG(ERROR) << "node[" << channel_shuffle_node_name << "] already exists.";
                continue;
            }
            ir::Node* channel_shuffle_node = node_ret_pair.first;
            channel_shuffle_node->SetType(ir::Node::Type("pmx", "ChannelShuffle", 1));

            auto param_ref = graph_data->attrs.find(channel_shuffle_node->GetId());
            if (param_ref == graph_data->attrs.end()) {
                auto channel_shuffle_param = std::make_shared<ppl::nn::pmx::ChannelShuffleParam>();
                channel_shuffle_param->group = group;
                graph_data->attrs[channel_shuffle_node->GetId()] = channel_shuffle_param;
            } else {
                LOG(ERROR) << "Node " << channel_shuffle_node->GetName() << "param exist.";
                continue;
            }

            /** 2. replace ops with fused op **/
            std::vector<ir::Node*> to_delete_nodes{reshape1_node, trans_node, reshape2_node};
            std::vector<ir::Edge*> inputs{reshape1_input_edge};
            std::vector<ir::Edge*> outputs{reshape2_output_edge};
            if (fuse_concat) {
                to_delete_nodes.insert(to_delete_nodes.begin(), reshape1_prev_node);
                inputs.resize(reshape1_prev_node->GetInputCount());
                for (uint32_t i = 0; i < reshape1_prev_node->GetInputCount(); i++) {
                    inputs[i] = graph_topo->GetEdge(reshape1_prev_node->GetInput(i));
                }
            }
            if (fuse_split) {
                to_delete_nodes.push_back(reshape2_next_nodes[0]);
                outputs.resize(reshape2_next_nodes[0]->GetOutputCount());
                for (uint32_t i = 0; i < reshape2_next_nodes[0]->GetOutputCount(); i++) {
                    outputs[i] = graph_topo->GetEdge(reshape2_next_nodes[0]->GetOutput(i));
                }
            }
            if (fuse_slice) {
                to_delete_nodes.push_back(reshape2_next_nodes[0]);
                to_delete_nodes.push_back(reshape2_next_nodes[1]);
                outputs.resize(2);
                outputs[0] = graph_topo->GetEdge(reshape2_next_nodes[0]->GetOutput(0));
                outputs[1] = graph_topo->GetEdge(reshape2_next_nodes[1]->GetOutput(0));
            }
            if (ppl::common::RC_SUCCESS !=
                ReplaceSubgraphWithOneNode(options, to_delete_nodes, inputs, outputs, channel_shuffle_node)) {
                LOG(ERROR) << "Replace sequence nodes with node << " << channel_shuffle_node->GetName() << " failed.";
                graph_data->attrs.erase(channel_shuffle_node->GetId());
                graph_topo->DelNode(channel_shuffle_node->GetId());
                continue;
            }

            /** 3. create opt_kernel **/
            ArmOptKernel* opt_kernel = nullptr;
            if (ppl::common::RC_SUCCESS != CreateArmOptKernel(options, channel_shuffle_node, &opt_kernel)) {
                LOG(ERROR) << "Node " << channel_shuffle_node->GetName() << "param exist.";
                graph_data->attrs.erase(channel_shuffle_node->GetId());
                graph_topo->DelNode(channel_shuffle_node->GetId());
                continue;
            }

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
            LOG(INFO) << "Successfully fused " << channel_shuffle_node_name;
#endif
            graph_changed = true;
        }
    }

    return graph_changed;
}

}}} // namespace ppl::nn::arm
