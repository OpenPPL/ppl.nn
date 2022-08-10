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
    auto tensor = graph->topo->GetEdge(edge_id);
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
            auto bn_input_edge = graph->topo->GetEdge(node->GetInput(0));
            auto bn_output_edge = graph->topo->GetEdge(node->GetOutput(0));

            if (bn_node->GetOutputCount() > 1) { // training phase BN, not to fuse
                continue;
            }
            if (bn_node->GetInputCount() != 5) {
                continue;
            }
            if (bn_input_edge->CalcConsumerCount() != 1 || IsGraphOutput(graph, bn_input_edge->GetId())) {
                continue;
            }

            auto predecessor_node = graph->topo->GetNode(bn_input_edge->GetProducer());
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
            auto conv_filter_edge = graph->topo->GetEdge(conv_node->GetInput(1));
            auto conv_bias_edge =
                conv_node->GetInputCount() > 2 ? graph->topo->GetEdge(conv_node->GetInput(2)) : nullptr;
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
            float* conv_filter_ptr = (float*)constants[conv_filter_edge->GetId()].data.GetData();
            float* conv_bias_ptr = nullptr;
            if (conv_bias_edge) {
                conv_bias_ptr = (float*)constants[conv_bias_edge->GetId()].data.GetData();
            } else { // if conv node has no bias, add bias tensor
                auto add_bias_edge_name = conv_node->GetName() + "_bias";
                auto edge_ret_pair = graph->topo->AddEdge(add_bias_edge_name);
                if (!edge_ret_pair.second) {
                    LOG(ERROR) << "edge[" << add_bias_edge_name << "] already exists.";
                    continue;
                }
                conv_bias_edge = edge_ret_pair.first;
                graph->topo->MarkAsConstant(conv_bias_edge->GetId());
                conv_node->AddInput(conv_bias_edge->GetId());
                conv_bias_edge->AddConsumer(conv_node->GetId());

                ir::Constant bias_constant;
                bias_constant.data.Resize(channels * sizeof(float), 0); // init bias to 0
                constants.emplace(conv_bias_edge->GetId(), bias_constant);
                conv_bias_ptr = (float*)constants[conv_bias_edge->GetId()].data.GetData();

                ir::Shape bias_shape;
                bias_shape.data_type = DATATYPE_FLOAT32;
                bias_shape.data_format = DATAFORMAT_NDARRAY;
                bias_shape.dims.resize(1, channels);
                shapes.emplace(conv_bias_edge->GetId(), bias_shape);
            }

            const float* bn_scale_ptr = (const float*)constants[bn_node->GetInput(1)].data.GetData();
            const float* bn_bias_ptr = (const float*)constants[bn_node->GetInput(2)].data.GetData();
            const float* bn_mean_ptr = (const float*)constants[bn_node->GetInput(3)].data.GetData();
            const float* bn_var_ptr = (const float*)constants[bn_node->GetInput(4)].data.GetData();

            float eps = 1e-5;
            if (attrs.find(bn_node->GetId()) != attrs.end()) {
                const ppl::nn::onnx::BatchNormalizationParam* param =
                    (const ppl::nn::onnx::BatchNormalizationParam*)attrs[bn_node->GetId()].get();
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
                auto initializer_edge = graph->topo->GetEdge(bn_node->GetInput(i));
                initializer_edge->DelConsumer(bn_node->GetId());
                if (initializer_edge->CalcConsumerCount() == 0 && !IsGraphOutput(graph, initializer_edge->GetId())) {
                    constants.erase(initializer_edge->GetId());
                    graph->topo->DelEdge(initializer_edge->GetId());
                }
            }

            graph->topo->DelNode(bn_node->GetId());
            graph->topo->DelEdge(bn_input_edge->GetId());

            graph_changed = true;
        }
    }

    return graph_changed;
}

// fuse convtranspose & batchnormalization
static bool FuseConvTransposeBatchNormalization(ir::Graph* graph) {
    bool graph_changed = false;

    auto& attrs = graph->data->attrs;
    auto& constants = graph->data->constants;
    auto& shapes = graph->data->shapes;

    for (auto it = graph->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "BatchNormalization") {
            // check topo pattern
            auto bn_node = node;
            auto bn_input_edge = graph->topo->GetEdge(node->GetInput(0));
            auto bn_output_edge = graph->topo->GetEdge(node->GetOutput(0));

            if (bn_node->GetOutputCount() > 1) { // training phase BN, not to fuse
                continue;
            }
            if (bn_node->GetInputCount() != 5) {
                continue;
            }
            if (bn_input_edge->CalcConsumerCount() != 1 || IsGraphOutput(graph, bn_input_edge->GetId())) {
                continue;
            }

            auto predecessor_node = graph->topo->GetNode(bn_input_edge->GetProducer());
            if (predecessor_node == nullptr || predecessor_node->GetType().domain != "" ||
                predecessor_node->GetType().name != "ConvTranspose") {
                continue;
            }
            auto convtranspose_node = predecessor_node;

            // check if related weights are all initializer, has valid shape and are fp32 ndarray
            bool all_initializer_and_fp32_ndarray = true;
            for (uint32_t i = 1; i < bn_node->GetInputCount(); i++) {
                if (!IsInitializerAndFp32NDArray(graph, bn_node->GetInput(i))) {
                    all_initializer_and_fp32_ndarray = false;
                    break;
                }
            }
            for (uint32_t i = 1; i < convtranspose_node->GetInputCount(); i++) {
                if (!IsInitializerAndFp32NDArray(graph, convtranspose_node->GetInput(i))) {
                    all_initializer_and_fp32_ndarray = false;
                    break;
                }
            }
            if (!all_initializer_and_fp32_ndarray) {
                continue;
            }

            // check if convtranspose is convtranspose2d
            auto convtranspose_filter_edge = graph->topo->GetEdge(convtranspose_node->GetInput(1));
            auto convtranspose_bias_edge =
                convtranspose_node->GetInputCount() > 2 ? graph->topo->GetEdge(convtranspose_node->GetInput(2)) : nullptr;
            const auto& convtranspose_filter_dims = shapes[convtranspose_filter_edge->GetId()].dims;
            if (convtranspose_filter_dims.size() != 4) { // not convtranspose2d
                continue;
            }

            // check if related weights have same channel num
            const uint32_t channels = convtranspose_filter_dims[1];
            bool all_same_channels = true;
            for (uint32_t i = 1; i < bn_node->GetInputCount(); i++) {
                if (shapes[bn_node->GetInput(i)].dims.size() != 1 || shapes[bn_node->GetInput(i)].dims[0] != channels) {
                    all_same_channels = false;
                    break;
                }
            }
            if (convtranspose_bias_edge &&
                (shapes[convtranspose_bias_edge->GetId()].dims.size() != 1 ||
                 shapes[convtranspose_bias_edge->GetId()].dims[0] != channels)) {
                all_same_channels = false;
            }
            if (!all_same_channels) {
                continue;
            }

            // all check passed, now fuse convtranspose & bn
            float* convtranspose_filter_ptr = (float*)constants[convtranspose_filter_edge->GetId()].data.GetData();
            float* convtranspose_bias_ptr = nullptr;
            if (convtranspose_bias_edge) {
                convtranspose_bias_ptr = (float*)constants[convtranspose_bias_edge->GetId()].data.GetData();
            } else { // if convtranspose node has no bias, add bias tensor
                auto add_bias_edge_name = convtranspose_node->GetName() + "_bias";
                auto edge_ret_pair = graph->topo->AddEdge(add_bias_edge_name);
                if (!edge_ret_pair.second) {
                    LOG(ERROR) << "edge[" << add_bias_edge_name << "] already exists.";
                    continue;
                }
                convtranspose_bias_edge = edge_ret_pair.first;
                graph->topo->MarkAsConstant(convtranspose_bias_edge->GetId());
                convtranspose_node->AddInput(convtranspose_bias_edge->GetId());
                convtranspose_bias_edge->AddConsumer(convtranspose_node->GetId());

                ir::Constant bias_constant;
                bias_constant.data.Resize(channels * sizeof(float), 0); // init bias to 0
                constants.emplace(convtranspose_bias_edge->GetId(), bias_constant);
                convtranspose_bias_ptr = (float*)constants[convtranspose_bias_edge->GetId()].data.GetData();

                ir::Shape bias_shape;
                bias_shape.data_type = DATATYPE_FLOAT32;
                bias_shape.data_format = DATAFORMAT_NDARRAY;
                bias_shape.dims.resize(1, channels);
                shapes.emplace(convtranspose_bias_edge->GetId(), bias_shape);
            }

            const float* bn_scale_ptr = (const float*)constants[bn_node->GetInput(1)].data.GetData();
            const float* bn_bias_ptr = (const float*)constants[bn_node->GetInput(2)].data.GetData();
            const float* bn_mean_ptr = (const float*)constants[bn_node->GetInput(3)].data.GetData();
            const float* bn_var_ptr = (const float*)constants[bn_node->GetInput(4)].data.GetData();

            float eps = 1e-5;
            if (attrs.find(bn_node->GetId()) != attrs.end()) {
                const ppl::nn::onnx::BatchNormalizationParam* param =
                    (const ppl::nn::onnx::BatchNormalizationParam*)attrs[bn_node->GetId()].get();
                eps = param->epsilon;
            }

            const int64_t nhw = convtranspose_filter_dims[0] * convtranspose_filter_dims[2] * convtranspose_filter_dims[3];
            const int64_t chw = convtranspose_filter_dims[1] * convtranspose_filter_dims[2] * convtranspose_filter_dims[3];
            const int64_t hw = convtranspose_filter_dims[2] * convtranspose_filter_dims[3];
            for (uint32_t c = 0; c < channels; c++) {
                // (x - mean) / sqrt(var + eps) * scale + bias -----> alpha * x + beta
                const float alpha = bn_scale_ptr[c] / sqrtf(bn_var_ptr[c] + eps);
                const float beta = bn_bias_ptr[c] - alpha * bn_mean_ptr[c];

                // alpha * (SUM(filter * x) + bias) + beta -----> SUM(alpha * filter * x) + alpha * bias
                // + beta
                for (int64_t i = 0; i < nhw; i++) {
                     convtranspose_filter_ptr[i / hw * chw + c * hw + i % hw] *= alpha;
                }
                convtranspose_bias_ptr[c] = alpha * convtranspose_bias_ptr[c] + beta;
            }

            // delete bn node's input & bn_node
            convtranspose_node->ReplaceOutput(bn_input_edge->GetId(), bn_output_edge->GetId());
            bn_output_edge->SetProducer(convtranspose_node->GetId());
            for (uint32_t i = 1; i < bn_node->GetInputCount(); i++) {
                auto initializer_edge = graph->topo->GetEdge(bn_node->GetInput(i));
                initializer_edge->DelConsumer(bn_node->GetId());
                if (initializer_edge->CalcConsumerCount() == 0 && !IsGraphOutput(graph, initializer_edge->GetId())) {
                    constants.erase(initializer_edge->GetId());
                    graph->topo->DelEdge(initializer_edge->GetId());
                }
            }

            graph->topo->DelNode(bn_node->GetId());
            graph->topo->DelEdge(bn_input_edge->GetId());

            graph_changed = true;
        }
    }

    return graph_changed;
}

RetCode FuseBNOptimizer::Optimize(ir::Graph* graph) const {
    while (FuseConvBatchNormalization(graph) || FuseConvTransposeBatchNormalization(graph));

    return RC_SUCCESS;
}

}} // namespace ppl::nn
