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

#include <math.h>
#include <string.h>

#include "ppl/nn/optimizers/fuse_bn_optimizer.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/params/onnx/batch_normalization_param.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

inline bool IsGraphOutput(const ir::Graph* graph, edgeid_t edge_id) {
    for (uint32_t i = 0; i < graph->topo->GetOutputCount(); i++) {
        if (graph->topo->GetOutput(i) == edge_id) {
            return true;
        }
    }
    return false;
}

inline bool IsInitializerAndFp32NDArray(const ir::Graph* graph, edgeid_t edge_id) {
    auto tensor = graph->topo->GetEdgeById(edge_id);
    if (!tensor || graph->data->constants.find(edge_id) == graph->data->constants.end() ||
        graph->data->shapes.find(edge_id) == graph->data->shapes.end() ||
        graph->data->shapes[edge_id].data_type != DATATYPE_FLOAT32 ||
        graph->data->shapes[edge_id].data_format != DATAFORMAT_NDARRAY) {
        return false;
    }
    return true;
}

// fuse conv & batchnormalization
static bool FuseConvBatchNormalization(ir::Graph* graph) {
    bool graph_changed = false;

    auto& attrs = graph->data->attrs;
    auto& constants = graph->data->constants;
    auto& shapes = graph->data->shapes;

    for (auto it = graph->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "BatchNormalization") {
            // check topo pattern
            auto bn_node = node;
            auto bn_input_edge = graph->topo->GetEdgeById(node->GetInput(0));
            auto bn_output_edge = graph->topo->GetEdgeById(node->GetOutput(0));

            if (bn_node->GetOutputCount() > 1) { // training phase BN, not to fuse
                continue;
            }
            if (bn_node->GetInputCount() != 5) {
                continue;
            }
            if (bn_input_edge->CalcConsumerCount() != 1 || IsGraphOutput(graph, bn_input_edge->GetId())) {
                continue;
            }

            auto predecessor_node = graph->topo->GetNodeById(bn_input_edge->GetProducer());
            if (predecessor_node == nullptr || predecessor_node->GetType().domain != "" ||
                predecessor_node->GetType().name != "Conv") {
                continue;
            }
            auto conv_node = predecessor_node;

            // check if related weights are all initializer, has valid shape and are fp32 ndarray
            bool all_initializer_and_fp32_ndarray = true;
            for (uint32_t i = 1; i < bn_node->GetInputCount(); i++) {
                if (!IsInitializerAndFp32NDArray(graph, bn_node->GetInput(i))) {
                    all_initializer_and_fp32_ndarray = false;
                    break;
                }
            }
            for (uint32_t i = 1; i < conv_node->GetInputCount(); i++) {
                if (!IsInitializerAndFp32NDArray(graph, conv_node->GetInput(i))) {
                    all_initializer_and_fp32_ndarray = false;
                    break;
                }
            }
            if (!all_initializer_and_fp32_ndarray) {
                continue;
            }

            // check if conv is conv2d
            auto conv_filter_edge = graph->topo->GetEdgeById(conv_node->GetInput(1));
            auto conv_bias_edge =
                conv_node->GetInputCount() > 2 ? graph->topo->GetEdgeById(conv_node->GetInput(2)) : nullptr;
            const auto& conv_filter_dims = shapes[conv_filter_edge->GetId()].dims;
            if (conv_filter_dims.size() != 4) { // not conv2d
                continue;
            }

            // check if related weights have same channel num
            const uint32_t channels = conv_filter_dims[0];
            bool all_same_channels = true;
            for (uint32_t i = 1; i < bn_node->GetInputCount(); i++) {
                if (shapes[bn_node->GetInput(i)].dims.size() != 1 || shapes[bn_node->GetInput(i)].dims[0] != channels) {
                    all_same_channels = false;
                    break;
                }
            }
            if (conv_bias_edge &&
                (shapes[conv_bias_edge->GetId()].dims.size() != 1 ||
                 shapes[conv_bias_edge->GetId()].dims[0] != channels)) {
                all_same_channels = false;
            }
            if (!all_same_channels) {
                continue;
            }

            // all check passed, now fuse conv & bn
            float* conv_filter_ptr = (float*)constants[conv_filter_edge->GetId()].data.data();
            float* conv_bias_ptr = nullptr;
            if (conv_bias_edge) {
                conv_bias_ptr = (float*)constants[conv_bias_edge->GetId()].data.data();
            } else { // if conv node has no bias, add bias tensor
                auto add_bias_edge_name = conv_node->GetName() + "_" + "bias";
                auto edge_ret_pair = graph->topo->AddEdge(add_bias_edge_name);
                if (!edge_ret_pair.second) {
                    LOG(ERROR) << "edge[" << add_bias_edge_name << "] already exists.";
                    continue;
                }
                conv_bias_edge = edge_ret_pair.first;
                conv_node->AddInput(conv_bias_edge->GetId());
                conv_bias_edge->AddConsumer(conv_node->GetId());

                ir::Constant bias_constant;
                bias_constant.data.resize(channels * sizeof(float), 0); // init bias to 0
                constants.emplace(conv_bias_edge->GetId(), bias_constant);
                conv_bias_ptr = (float*)constants[conv_bias_edge->GetId()].data.data();

                ir::Shape bias_shape;
                bias_shape.data_type = DATATYPE_FLOAT32;
                bias_shape.data_format = DATAFORMAT_NDARRAY;
                bias_shape.dims.resize(1, channels);
                shapes.emplace(conv_bias_edge->GetId(), bias_shape);
            }

            const float* bn_scale_ptr = (const float*)constants[bn_node->GetInput(1)].data.data();
            const float* bn_bias_ptr = (const float*)constants[bn_node->GetInput(2)].data.data();
            const float* bn_mean_ptr = (const float*)constants[bn_node->GetInput(3)].data.data();
            const float* bn_var_ptr = (const float*)constants[bn_node->GetInput(4)].data.data();

            float eps = 1e-5;
            if (attrs.find(bn_node->GetId()) != attrs.end()) {
                const ppl::nn::common::BatchNormalizationParam* param =
                    (const ppl::nn::common::BatchNormalizationParam*)attrs[bn_node->GetId()].get();
                eps = param->epsilon;
            }

            const int64_t chw = conv_filter_dims[1] * conv_filter_dims[2] * conv_filter_dims[3];
            for (uint32_t c = 0; c < channels; c++) {
                // (x - mean) / sqrt(var + eps) * scale + bias -----> alpha * x + beta
                const float alpha = bn_scale_ptr[c] / sqrtf(bn_var_ptr[c] + eps);
                const float beta = bn_bias_ptr[c] - alpha * bn_mean_ptr[c];

                // alpha * (SUM(filter * x) + bias) + beta -----> SUM(alpha * filter * x) + alpha * bias
                // + beta
                for (int64_t i = 0; i < chw; i++) {
                    conv_filter_ptr[c * chw + i] *= alpha;
                }
                conv_bias_ptr[c] = alpha * conv_bias_ptr[c] + beta;
            }

            // delete bn node's input & bn_node
            conv_node->ReplaceOutput(bn_input_edge->GetId(), bn_output_edge->GetId());
            bn_output_edge->SetProducer(conv_node->GetId());
            for (uint32_t i = 1; i < bn_node->GetInputCount(); i++) {
                auto initializer_edge = graph->topo->GetEdgeById(bn_node->GetInput(i));
                initializer_edge->DelConsumer(bn_node->GetId());
                if (initializer_edge->CalcConsumerCount() == 0 && !IsGraphOutput(graph, initializer_edge->GetId())) {
                    constants.erase(initializer_edge->GetId());
                    graph->topo->DelEdgeById(initializer_edge->GetId());
                }
            }

            graph->topo->DelNodeById(bn_node->GetId());
            graph->topo->DelEdgeById(bn_input_edge->GetId());

            graph_changed = true;
        }
    }

    return graph_changed;
}

// fuse conv & mul
static RetCode FuseConvMul(ir::Graph* graph) {
    bool graph_changed = false;

    auto& constants = graph->data->constants;
    auto& shapes = graph->data->shapes;

    for (auto it = graph->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "Conv") {
            // check topo pattern
            auto conv_node = node;
            auto conv_filter_edge = graph->topo->GetEdgeById(conv_node->GetInput(1));
            auto conv_bias_edge =
                conv_node->GetInputCount() >= 3 ? graph->topo->GetEdgeById(conv_node->GetInput(2)) : nullptr;
            auto conv_output_edge = graph->topo->GetEdgeById(conv_node->GetOutput(0));
            if (!conv_output_edge || conv_output_edge->CalcConsumerCount() != 1 ||
                IsGraphOutput(graph, conv_output_edge->GetId())) {
                continue;
            }

            auto successor = graph->topo->GetNodeById(conv_output_edge->CreateConsumerIter().Get());
            if (!successor || successor->GetType().domain != "" || successor->GetType().name != "Mul") {
                continue;
            }
            auto mul_node = successor;
            auto mul_output_edge = graph->topo->GetEdgeById(mul_node->GetOutput(0));
            if (!mul_output_edge || mul_output_edge->CalcConsumerCount() != 1 ||
                IsGraphOutput(graph, mul_output_edge->GetId())) {
                continue;
            }
            auto scale_edge = graph->topo->GetEdgeById(mul_node->GetInput(1));
            if (scale_edge == conv_output_edge) {
                scale_edge = graph->topo->GetEdgeById(mul_node->GetInput(0));
            }

            // check if related weights are all initializer, has valid shape and are fp32 ndarray
            bool all_initializer_and_fp32_ndarray = true;
            for (uint32_t i = 1; i < conv_node->GetInputCount(); i++) {
                if (!IsInitializerAndFp32NDArray(graph, conv_node->GetInput(i))) {
                    all_initializer_and_fp32_ndarray = false;
                    break;
                }
            }
            if (!IsInitializerAndFp32NDArray(graph, scale_edge->GetId())) {
                all_initializer_and_fp32_ndarray = false;
            }
            if (!all_initializer_and_fp32_ndarray) {
                continue;
            }

            // check if conv is conv2d
            const auto& conv_filter_dims = shapes[conv_filter_edge->GetId()].dims;
            if (conv_filter_dims.size() != 4) { // not conv2d
                continue;
            }
            const uint32_t channels = conv_filter_dims[0];
            if (conv_bias_edge &&
                (shapes[conv_bias_edge->GetId()].dims.size() != 1 ||
                 shapes[conv_bias_edge->GetId()].dims[0] != channels)) {
                continue;
            }

            // check scale's channel num
            const auto& scale_dims = shapes[scale_edge->GetId()].dims;
            if (scale_dims.size() > 4) {
                continue;
            }

            const int32_t scale_c_dim_idx = scale_dims.size() >= 3 ? scale_dims.size() - 3 : -1;
            bool invalid_scale_dims = false;
            for (int32_t i = 0; i < (int32_t)scale_dims.size(); i++) {
                if (i == scale_c_dim_idx) {
                    if (scale_dims[i] != channels && scale_dims[i] != 1) {
                        invalid_scale_dims = true;
                        break;
                    }
                } else {
                    if (scale_dims[i] != 1) {
                        invalid_scale_dims = true;
                        break;
                    }
                }
            }
            if (invalid_scale_dims) {
                continue;
            }

            // all check passed, now first process scale
            std::vector<float> scale_data(channels);
            int64_t scale_num_elements = 1;
            for (uint32_t i = 0; i < scale_dims.size(); i++) {
                scale_num_elements *= scale_dims[i];
            }
            const float* scale_ori_data = (const float*)constants[scale_edge->GetId()].data.data();
            if (scale_num_elements == channels) {
                memcpy(scale_data.data(), scale_ori_data, scale_num_elements * sizeof(float));
            } else {
                for (int64_t i = 0; i < channels; i++) {
                    scale_data[i] = scale_ori_data[0];
                }
            }

            // fuse conv & mul
            float* conv_filter_ptr = (float*)constants[conv_filter_edge->GetId()].data.data();
            float* conv_bias_ptr = nullptr;
            if (conv_bias_edge) {
                conv_bias_ptr = (float*)constants[conv_bias_edge->GetId()].data.data();
            }

            const int64_t chw = conv_filter_dims[1] * conv_filter_dims[2] * conv_filter_dims[3];
            for (uint32_t c = 0; c < channels; c++) {
                // scale * (SUM(filter * x) + bias) -----> SUM(scale * filter * x) + scale * bias
                for (int64_t i = 0; i < chw; i++) {
                    conv_filter_ptr[c * chw + i] *= scale_data[c];
                }
                if (conv_bias_ptr) {
                    conv_bias_ptr[c] = scale_data[c] * conv_bias_ptr[c];
                }
            }

            // delete mul node and related edge
            conv_node->ReplaceOutput(conv_output_edge->GetId(), mul_output_edge->GetId());
            mul_output_edge->SetProducer(conv_node->GetId());
            scale_edge->DelConsumer(mul_node->GetId());

            if (scale_edge->CalcConsumerCount() == 0 && !IsGraphOutput(graph, scale_edge->GetId())) {
                constants.erase(scale_edge->GetId());
                graph->topo->DelEdgeById(scale_edge->GetId());
            }
            graph->topo->DelEdgeById(conv_output_edge->GetId());
            graph->topo->DelNodeById(mul_node->GetId());

            graph_changed = true;
        }
    }

    return graph_changed;
}

// fuse conv & add
static RetCode FuseConvAdd(ir::Graph* graph) {
    bool graph_changed = false;

    auto& constants = graph->data->constants;
    auto& shapes = graph->data->shapes;

    for (auto it = graph->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "Conv") {
            // check topo pattern
            auto conv_node = node;
            auto conv_filter_edge = graph->topo->GetEdgeById(conv_node->GetInput(1));
            auto conv_bias_edge =
                conv_node->GetInputCount() >= 3 ? graph->topo->GetEdgeById(conv_node->GetInput(2)) : nullptr;
            auto conv_output_edge = graph->topo->GetEdgeById(conv_node->GetOutput(0));
            if (!conv_output_edge || conv_output_edge->CalcConsumerCount() != 1 ||
                IsGraphOutput(graph, conv_output_edge->GetId())) {
                continue;
            }

            auto successor = graph->topo->GetNodeById(conv_output_edge->CreateConsumerIter().Get());
            if (!successor || successor->GetType().domain != "" || successor->GetType().name != "Add") {
                continue;
            }
            auto add_node = successor;
            auto add_output_edge = graph->topo->GetEdgeById(add_node->GetOutput(0));
            if (!add_output_edge || add_output_edge->CalcConsumerCount() != 1 ||
                IsGraphOutput(graph, add_output_edge->GetId())) {
                continue;
            }
            auto shift_edge = graph->topo->GetEdgeById(add_node->GetInput(1));
            if (shift_edge == conv_output_edge) {
                shift_edge = graph->topo->GetEdgeById(add_node->GetInput(0));
            }

            // check if related weights are all initializer, has valid shape and are fp32 ndarray
            bool all_initializer_and_fp32_ndarray = true;
            for (uint32_t i = 1; i < conv_node->GetInputCount(); i++) {
                if (!IsInitializerAndFp32NDArray(graph, conv_node->GetInput(i))) {
                    all_initializer_and_fp32_ndarray = false;
                    break;
                }
            }
            if (!IsInitializerAndFp32NDArray(graph, shift_edge->GetId())) {
                all_initializer_and_fp32_ndarray = false;
            }
            if (!all_initializer_and_fp32_ndarray) {
                continue;
            }

            // check if conv is conv2d
            const auto& conv_filter_dims = shapes[conv_filter_edge->GetId()].dims;
            if (conv_filter_dims.size() != 4) { // not conv2d
                continue;
            }
            const uint32_t channels = conv_filter_dims[0];
            if (conv_bias_edge &&
                (shapes[conv_bias_edge->GetId()].dims.size() != 1 ||
                 shapes[conv_bias_edge->GetId()].dims[0] != channels)) {
                continue;
            }

            // check shift's channel num
            const auto& shift_dims = shapes[shift_edge->GetId()].dims;
            if (shift_dims.size() > 4) {
                continue;
            }

            const int32_t shift_c_dim_idx = shift_dims.size() >= 3 ? shift_dims.size() - 3 : -1;
            bool invalid_shift_dims = false;
            for (int32_t i = 0; i < (int32_t)shift_dims.size(); i++) {
                if (i == shift_c_dim_idx) {
                    if (shift_dims[i] != channels && shift_dims[i] != 1) {
                        invalid_shift_dims = true;
                        break;
                    }
                } else {
                    if (shift_dims[i] != 1) {
                        invalid_shift_dims = true;
                        break;
                    }
                }
            }
            if (invalid_shift_dims) {
                continue;
            }

            // all check passed, now first process shift
            std::vector<float> shift_data(channels);
            int64_t shift_num_elements = 1;
            for (uint32_t i = 0; i < shift_dims.size(); i++) {
                shift_num_elements *= shift_dims[i];
            }
            const float* shift_ori_data = (const float*)constants[shift_edge->GetId()].data.data();
            if (shift_num_elements == channels) {
                memcpy(shift_data.data(), shift_ori_data, shift_num_elements * sizeof(float));
            } else {
                for (int64_t i = 0; i < channels; i++) {
                    shift_data[i] = shift_ori_data[0];
                }
            }

            // fuse conv & add
            float* conv_bias_ptr = nullptr;
            if (conv_bias_edge) {
                conv_bias_ptr = (float*)constants[conv_bias_edge->GetId()].data.data();
            } else { // if conv node has no bias, add bias tensor
                auto add_bias_edge_name = conv_node->GetName() + "_" + "bias";
                auto edge_ret_pair = graph->topo->AddEdge(add_bias_edge_name);
                if (!edge_ret_pair.second) {
                    LOG(ERROR) << "edge[" << add_bias_edge_name << "] already exists.";
                    continue;
                }
                conv_bias_edge = edge_ret_pair.first;
                conv_node->AddInput(conv_bias_edge->GetId());
                conv_bias_edge->AddConsumer(conv_node->GetId());

                ir::Constant bias_constant;
                bias_constant.data.resize(channels * sizeof(float), 0); // init bias to 0
                constants.emplace(conv_bias_edge->GetId(), bias_constant);
                conv_bias_ptr = (float*)constants[conv_bias_edge->GetId()].data.data();

                ir::Shape bias_shape;
                bias_shape.data_type = DATATYPE_FLOAT32;
                bias_shape.data_format = DATAFORMAT_NDARRAY;
                bias_shape.dims.resize(1, channels);
                shapes.emplace(conv_bias_edge->GetId(), bias_shape);
            }

            for (uint32_t c = 0; c < channels; c++) {
                // SUM(filter * x) + bias + shift
                conv_bias_ptr[c] = conv_bias_ptr[c] + shift_data[c];
            }

            // delete add node and related edge
            conv_node->ReplaceOutput(conv_output_edge->GetId(), add_output_edge->GetId());
            add_output_edge->SetProducer(conv_node->GetId());
            shift_edge->DelConsumer(add_node->GetId());

            if (shift_edge->CalcConsumerCount() == 0 && !IsGraphOutput(graph, shift_edge->GetId())) {
                constants.erase(shift_edge->GetId());
                graph->topo->DelEdgeById(shift_edge->GetId());
            }
            graph->topo->DelEdgeById(conv_output_edge->GetId());
            graph->topo->DelNodeById(add_node->GetId());

            graph_changed = true;
        }
    }

    return graph_changed;
}

RetCode FuseBNOptimizer::Optimize(ir::Graph* graph) const {
    while (FuseConvBatchNormalization(graph) || FuseConvMul(graph) || FuseConvAdd(graph))
        ;

    return RC_SUCCESS;
}

}} // namespace ppl::nn
