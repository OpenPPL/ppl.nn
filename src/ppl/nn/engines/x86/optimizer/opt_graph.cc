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

#include "ppl/nn/engines/x86/optimizer/opt_graph.h"
#include "ppl/nn/engines/x86/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/conv_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/ppl/channel_shuffle_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/add_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/mul_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/sub_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/div_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/gemm_op.h"
#include "ppl/nn/engines/utils.h"
#include "ppl/nn/params/onnx/transpose_param.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/batch_normalization_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/ppl/swish_op.h"
#include <string.h>

//#define SHOW_GRAPH_VIS
#ifdef SHOW_GRAPH_VIS
#include "ppl/nn/auxtools/to_graphviz.h"
#include <fstream>
#endif
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode OptGraph::InitKernels(const ir::Graph* graph) {
    auto topo = graph->topo.get();
    for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        auto& type = node->GetType();
        auto creator = OptKernelCreatorManager::Instance()->Find(type.domain, type.name);
        if (!creator) {
            LOG(ERROR) << "cannot find creator for X86OptKernel[" << node->GetName() << "] type[" << type.domain << ":"
                       << type.name << "]";
            return RC_NOT_FOUND;
        }

        auto opt_kernel = unique_ptr<X86OptKernel>(creator(node));
        if (!opt_kernel) {
            LOG(ERROR) << "create X86OptKernel failed: oom";
            return RC_OUT_OF_MEMORY;
        }

        info_->kernels.emplace(node->GetId(), std::move(opt_kernel));
    }

    return RC_SUCCESS;
}

RetCode OptGraph::InitTensorImpls() {
    tensor_impls_.clear();
    auto& shapes = graph_->data->shapes;
    for (auto it = graph_->topo->CreateEdgeIter(); it->IsValid(); it->Forward()) {
        auto edge = it->Get();
        auto edge_id = edge->GetId();
        auto tensor_type = graph_->data->constants.find(edge_id) == graph_->data->constants.end() ? TENSORTYPE_NORMAL
                                                                                                  : TENSORTYPE_RESERVED;
        TensorImpl* tensor = new TensorImpl(edge, tensor_type);
        if (shapes.find(edge_id) != shapes.end()) {
            utils::IrShape2TensorShape(shapes[edge_id], &tensor->GetShape());
        } else {
            tensor->GetShape().SetDataFormat(DATAFORMAT_NDARRAY);
        }
        tensor_impls_.emplace(edge_id, unique_ptr<TensorImpl>(tensor));
    }
    return RC_SUCCESS;
}

RetCode OptGraph::Init(ir::Graph* graph, utils::SharedResource* resource, RuntimePartitionInfo* info) {
    resource_ = resource;
    graph_ = graph;
    info_ = info;

    auto status = InitKernels(graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init kernels failed: " << GetRetCodeStr(status);
        return status;
    }

    status = InitTensorImpls();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init tensor impls failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

#define REORDER_INPUT 0
#define REORDER_OUTPUT 1
#define REORDER_EXTRA_INPUT 2

RetCode OptGraph::AddReorderOp(const OptKernelOptions& options, const edgeid_t& edge_id, const nodeid_t& node_id,
                               const int32_t& reorder_type, const ppl::common::dataformat_t& reorder_in_format,
                               const ppl::common::dataformat_t& reorder_out_format) {
    auto edge = graph_->topo->GetEdgeById(edge_id);
    auto node = graph_->topo->GetNodeById(node_id);

    std::string reorder_node_name = "";
    if (reorder_type == REORDER_INPUT) {
        reorder_node_name = "ReorderInput_" + edge->GetName() + "_of_" + node->GetName();
    } else if (reorder_type == REORDER_OUTPUT) {
        reorder_node_name = "ReorderOutput_" + edge->GetName() + "_of_" + node->GetName();
    } else if (reorder_type == REORDER_EXTRA_INPUT) {
        reorder_node_name = "ReorderExtraInput_" + edge->GetName() + "_of_" + node->GetName();
    }

    auto node_ret_pair = graph_->topo->AddNode(reorder_node_name);
    if (!node_ret_pair.second) {
        LOG(ERROR) << "node[" << reorder_node_name << "] already exists.";
        return RC_EXISTS;
    }
    ir::Node* reorder_node = node_ret_pair.first; // TODO: change name for easy to understand
    reorder_node->SetType(ir::Node::Type("ppl", "Reorder"));

    std::string reorder_edge_name = reorder_node_name + "_edge";
    auto edge_ret_pair = graph_->topo->AddEdge(reorder_edge_name);
    if (!edge_ret_pair.second) {
        LOG(ERROR) << "edge[" << reorder_edge_name << "] already exists.";
        return RC_EXISTS;
    }
    ir::Edge* reorder_edge = edge_ret_pair.first;

    if (reorder_type == REORDER_INPUT ||
        reorder_type == REORDER_EXTRA_INPUT) { // edge -> reorder_node -> reorder_edge -> node
        reorder_node->AddInput(edge_id);
        reorder_node->AddOutput(reorder_edge->GetId());
        reorder_edge->SetProducer(reorder_node->GetId());
        reorder_edge->AddConsumer(node_id);

        edge->DelConsumer(node_id);
        edge->AddConsumer(reorder_node->GetId());
        if (reorder_type == REORDER_INPUT) {
            node->ReplaceInput(edge_id, reorder_edge->GetId());
        } else if (reorder_type == REORDER_EXTRA_INPUT) {
            node->ReplaceExtraInput(edge_id, reorder_edge->GetId());
        }
    } else if (reorder_type == REORDER_OUTPUT) { // node -> reorder_edge -> reorder_node ->  edge
        reorder_node->AddInput(reorder_edge->GetId());
        reorder_node->AddOutput(edge_id);
        reorder_edge->SetProducer(node_id);
        reorder_edge->AddConsumer(reorder_node->GetId());

        edge->SetProducer(reorder_node->GetId());
        node->ReplaceOutput(edge_id, reorder_edge->GetId());
    }

    auto type = reorder_node->GetType();
    auto creator = OptKernelCreatorManager::Instance()->Find(type.domain, type.name);
    if (!creator) {
        LOG(ERROR) << "cannot find creator for X86OptKernel[" << reorder_node->GetName() << "] type[" << type.domain
                   << ":" << type.name << "]";
        return RC_NOT_FOUND;
    }

    auto opt_kernel = unique_ptr<X86OptKernel>(creator(reorder_node));
    if (!opt_kernel) {
        LOG(ERROR) << "create X86OptKernel failed: oom";
        return RC_OUT_OF_MEMORY;
    }

    auto status = opt_kernel->Init(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Init for kernel[" << opt_kernel->GetNode()->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }
    opt_kernel->SetOutputDataFormat(0, reorder_out_format);
    info_->kernels.emplace(reorder_node->GetId(), std::move(opt_kernel));

    TensorImpl* tensor = new TensorImpl(reorder_edge, TENSORTYPE_NORMAL);
    tensor->GetShape().SetDataFormat((reorder_type == REORDER_INPUT || reorder_type == REORDER_EXTRA_INPUT)
                                         ? reorder_out_format
                                         : reorder_in_format);
    tensor_impls_.emplace(reorder_edge->GetId(), unique_ptr<TensorImpl>(tensor));

    // LOG(INFO) << "successfully add reorder op " << reorder_node_name << " to reorder " <<
    // GetDataFormatStr(reorder_in_format) << " to " << GetDataFormatStr(reorder_out_format) << ".";
    return RC_SUCCESS;
}

inline bool IsGraphOutput(const ir::Graph* graph, edgeid_t edge_id) {
    for (uint32_t i = 0; i < graph->topo->GetOutputCount(); i++) {
        if (graph->topo->GetOutput(i) == edge_id) {
            return true;
        }
    }
    return false;
}

RetCode OptGraph::LayoutOptimize(const OptKernelOptions& options) {
    vector<nodeid_t> sorted_nodes;
    graph_->topo->TopologicalSort([&sorted_nodes](nodeid_t nid) -> void {
        sorted_nodes.push_back(nid);
    });

    for (auto node_id : sorted_nodes) {
        if (info_->kernels.find(node_id) == info_->kernels.end()) {
            LOG(ERROR) << "cannot find node_id " << node_id << " in RuntimePartitionInfo.";
            return RC_NOT_FOUND;
        }
        auto kernel = (X86OptKernel*)info_->kernels[node_id].get();
        auto node = kernel->GetNode();

        InputOutputInfo IOinfo;
        IOinfo.SetNode(node);
        IOinfo.SetAcquireObjectFunc([this](edgeid_t eid, uint32_t, Device*) -> EdgeObject* {
            auto iter = tensor_impls_.find(eid);
            if (iter == tensor_impls_.end()) {
                return nullptr;
            }
            return iter->second.get();
        });

        auto status = kernel->SelectAlgorithm(IOinfo, options);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "kernel[" << node->GetName() << "] SelectAlgorithm failed: " << GetRetCodeStr(status);
            return status;
        }

        vector<dataformat_t> selected_input_formats(node->GetInputCount(), DATAFORMAT_NDARRAY);
        vector<dataformat_t> selected_output_formats(node->GetOutputCount(), DATAFORMAT_NDARRAY);

        status = kernel->SelectFormat(IOinfo, &selected_input_formats, &selected_output_formats);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "kernel[" << node->GetName() << "] SelectFormat failed: " << GetRetCodeStr(status);
            return status;
        }

        for (uint32_t i = 0; i < node->GetInputCount(); i++) {
            auto edge_id = node->GetInput(i);
            if (edge_id == INVALID_EDGEID) {
                continue;
            }
            auto input_format = tensor_impls_[edge_id]->GetShape().GetDataFormat();
            auto selected_input_format = selected_input_formats[i];
            if (input_format != selected_input_format) {
                status = AddReorderOp(options, edge_id, node_id, REORDER_INPUT, input_format, selected_input_format);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "add reorder op failed.";
                    return status;
                }
            }
        }

        // extra input(used by if/loop op) force to be ndarray
        for (uint32_t i = 0; i < node->GetExtraInputCount(); i++) {
            auto edge_id = node->GetExtraInput(i);
            auto extra_input_format = tensor_impls_[edge_id]->GetShape().GetDataFormat();
            if (extra_input_format != ppl::common::DATAFORMAT_NDARRAY) {
                status = AddReorderOp(options, edge_id, node_id, REORDER_EXTRA_INPUT, extra_input_format,
                                      ppl::common::DATAFORMAT_NDARRAY);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "add reorder op failed.";
                    return status;
                }
            }
        }

        for (uint32_t i = 0; i < node->GetOutputCount(); i++) {
            auto edge_id = node->GetOutput(i);
            auto selected_output_format = selected_output_formats[i];
            tensor_impls_[edge_id]->GetShape().SetDataFormat(selected_output_format);
            kernel->SetOutputDataFormat(i, selected_output_format);
        }
    }

    auto status = FuseReorderOp();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "FuseReorderOp failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode OptGraph::FuseReorderOp() {
    vector<nodeid_t> sorted_nodes;
    graph_->topo->TopologicalSort([&sorted_nodes](nodeid_t nid) -> void {
        sorted_nodes.push_back(nid);
    });

    for (auto node_it = graph_->topo->CreateNodeIter(); node_it->IsValid(); node_it->Forward()) {
        auto node = node_it->Get();
        if (node->GetType().domain == "ppl" && node->GetType().name == "Reorder") {
            auto input_edge_id = node->GetInput(0);
            auto input_edge = graph_->topo->GetEdgeById(input_edge_id);

            // find reorder op group with same output_format so that can be merged together
            std::map<dataformat_t, std::vector<ir::Node*>> reorder_op_groups;
            for (auto it = input_edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
                auto consumer_node = graph_->topo->GetNodeById(it.Get());
                if (consumer_node->GetType().domain == "ppl" && consumer_node->GetType().name == "Reorder") {
                    auto output_edge_id = consumer_node->GetOutput(0);
                    auto output_format = tensor_impls_[output_edge_id]->GetShape().GetDataFormat();

                    if (reorder_op_groups.find(output_format) == reorder_op_groups.end()) {
                        reorder_op_groups.emplace(output_format, std::vector<ir::Node*>(0));
                    }
                    reorder_op_groups[output_format].push_back(consumer_node);
                }
            }

            // merge reorder op group into one reorder op
            for (auto it = reorder_op_groups.begin(); it != reorder_op_groups.end(); ++it) {
                auto op_group = it->second;
                if (op_group.size() <= 1) {
                    continue;
                }

                // merge op_group[1:] into op_group[0]
                // input_edge -> op_group[0]           -> merged_output_edge -> consumer_node_0
                //            ...
                //            -> op_group[i](del_node) -> del_output_edge    -> consumer_node_i
                auto merged_output_edge_id = op_group[0]->GetOutput(0);
                auto merged_output_edge = graph_->topo->GetEdgeById(merged_output_edge_id);
                for (uint32_t i = 1; i < op_group.size(); i++) {
                    auto del_node_id = op_group[i]->GetId();
                    auto del_output_edge_id = op_group[i]->GetOutput(0);
                    auto del_output_edge = graph_->topo->GetEdgeById(del_output_edge_id);
                    auto consumer_node_id = del_output_edge->CreateConsumerIter().Get(); // only has one consumer
                    auto consumer_node = graph_->topo->GetNodeById(consumer_node_id);

                    merged_output_edge->AddConsumer(consumer_node_id);
                    consumer_node->ReplaceInput(del_output_edge_id, merged_output_edge_id);
                    input_edge->DelConsumer(del_node_id);

                    // LOG(INFO) << "kernel " << op_group[i]->GetName() << " is merged into kernel " <<
                    // op_group[0]->GetName() << ".";
                    info_->kernels.erase(del_node_id);
                    graph_->topo->DelNodeById(del_node_id);
                    graph_->topo->DelEdgeById(del_output_edge_id);
                }
            }
        }
    }

    return RC_SUCCESS;
}

static bool IsFloatReLU6(const ir::Graph* graph, const ir::Node* clip_node) {
    if (clip_node->GetType().domain != "" || clip_node->GetType().name != "Clip") {
        return false;
    }
    if (clip_node->GetInputCount() != 3) {
        return false;
    }

    auto min_edge_id = clip_node->GetInput(1);
    auto max_edge_id = clip_node->GetInput(2);
    auto& constants = graph->data->constants;
    if (constants.find(min_edge_id) == constants.end() || constants.find(max_edge_id) == constants.end()) {
        return false;
    }

    auto& shapes = graph->data->shapes;
    if (shapes.find(min_edge_id) == shapes.end() || shapes.find(max_edge_id) == shapes.end()) {
        return false;
    }
    if (shapes[min_edge_id].data_type != DATATYPE_FLOAT32 || shapes[max_edge_id].data_type != DATATYPE_FLOAT32) {
        return false;
    }

    float min_val = *((float*)constants[min_edge_id].data.data());
    float max_val = *((float*)constants[max_edge_id].data.data());
    if (min_val != 0.0f && max_val != 6.0f) {
        return false;
    }

    return true;
}

bool OptGraph::FuseConvActivation() {
    bool graph_changed = false;

    for (auto it = graph_->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "Conv") {
            auto conv_node = node;
            auto conv_output_edge_id = conv_node->GetOutput(0);
            auto conv_output_edge = graph_->topo->GetEdgeById(conv_output_edge_id);
            if (conv_output_edge->CalcConsumerCount() != 1) {
                continue;
            }
            if (IsGraphOutput(graph_, conv_output_edge_id)) {
                continue;
            }

            auto successor_node_id = conv_output_edge->CreateConsumerIter().Get();
            auto successor_node = graph_->topo->GetNodeById(successor_node_id);
            if (successor_node->GetType().domain != "") {
                continue;
            }

            auto conv_kernel = static_cast<ConvOp*>(info_->kernels[conv_node->GetId()].get());
            if (successor_node->GetType().name == "Relu") {
                if (!conv_kernel->SetFuseReLU()) { // set fuse flag to conv_op
                    continue;
                }
            } else if (IsFloatReLU6(graph_, successor_node)) {
                if (!conv_kernel->SetFuseReLU6()) { // set fuse flag to conv_op
                    continue;
                }
                // remove relu6's input min/max's connect in advance
                auto min_edge = graph_->topo->GetEdgeById(successor_node->GetInput(1));
                auto max_edge = graph_->topo->GetEdgeById(successor_node->GetInput(2));
                min_edge->DelConsumer(successor_node->GetId());
                max_edge->DelConsumer(successor_node->GetId());
                if (min_edge->CalcConsumerCount() == 0 && !IsGraphOutput(graph_, min_edge->GetId())) {
                    graph_->data->constants.erase(min_edge->GetId());
                    graph_->topo->DelEdgeById(min_edge->GetId());
                }
                if (max_edge->CalcConsumerCount() == 0 && !IsGraphOutput(graph_, max_edge->GetId())) {
                    graph_->data->constants.erase(max_edge->GetId());
                    graph_->topo->DelEdgeById(max_edge->GetId());
                }
            } else {
                continue;
            }

            auto activation_node = successor_node;
            auto activation_node_id = activation_node->GetId();
            auto activation_output_edge_id = activation_node->GetOutput(0);
            auto activation_output_edge = graph_->topo->GetEdgeById(activation_output_edge_id);
            // conv_node -> conv_output_edge -> activation_node -> activation_output_edge
            // conv_node                                        -> activation_output_edge
            conv_node->ReplaceOutput(conv_output_edge_id, activation_output_edge_id);
            activation_output_edge->SetProducer(conv_node->GetId());

            // LOG(INFO) << "merge kernel " << activation_node->GetName() << " into kernel " << conv_node->GetName() <<
            // ".";
            info_->kernels.erase(activation_node_id);
            graph_->topo->DelNodeById(activation_node_id);
            graph_->topo->DelEdgeById(conv_output_edge_id);

            graph_changed = true;
        }
    }

    return graph_changed;
}

RetCode OptGraph::TryToInferType(X86Device* device) {
    vector<nodeid_t> sorted_nodes;
    graph_->topo->TopologicalSort([&sorted_nodes](nodeid_t nid) -> void {
        sorted_nodes.push_back(nid);
    });

    for (auto node_id : sorted_nodes) {
        auto node = graph_->topo->GetNodeById(node_id);
        bool all_inputs_has_type = true;
        for (uint32_t i = 0; i < node->GetInputCount(); i++) {
            auto input_edge = graph_->topo->GetEdgeById(node->GetInput(i));
            if (!input_edge) { // some op may have emtpy input
                continue;
            }
            if (tensor_impls_.find(input_edge->GetId()) == tensor_impls_.end() ||
                tensor_impls_[input_edge->GetId()]->GetShape().GetDataType() == DATATYPE_UNKNOWN) {
                all_inputs_has_type = false;
                break;
            }
        }
        if (!all_inputs_has_type) {
            continue;
        }

        InputOutputInfo IOinfo;
        IOinfo.SetNode(node);
        IOinfo.SetAcquireObjectFunc([this](edgeid_t eid, uint32_t, Device*) -> EdgeObject* {
            auto iter = tensor_impls_.find(eid);
            if (iter == tensor_impls_.end()) {
                return nullptr;
            }
            return iter->second.get();
        });

        auto kernel = (X86OptKernel*)(info_->kernels[node_id].get());
        kernel->InferTypes(&IOinfo);
    }

    return RC_SUCCESS;
}

RetCode OptGraph::TryToInferDims(X86Device* device) {
    vector<nodeid_t> sorted_nodes;
    graph_->topo->TopologicalSort([&sorted_nodes](nodeid_t nid) -> void {
        sorted_nodes.push_back(nid);
    });

    for (auto node_id : sorted_nodes) {
        auto node = graph_->topo->GetNodeById(node_id);
        bool all_inputs_has_dims = true;
        for (uint32_t i = 0; i < node->GetInputCount(); i++) {
            auto input_edge = graph_->topo->GetEdgeById(node->GetInput(i));
            if (!input_edge) { // some op may have emtpy input
                continue;
            }
            if (tensor_impls_.find(input_edge->GetId()) == tensor_impls_.end() ||
                tensor_impls_[input_edge->GetId()]->GetShape().GetDimCount() == 0) {
                all_inputs_has_dims = false;
                break;
            }
        }
        if (!all_inputs_has_dims) {
            continue;
        }

        InputOutputInfo IOinfo;
        IOinfo.SetNode(node);
        IOinfo.SetAcquireObjectFunc([this](edgeid_t eid, uint32_t, Device*) -> EdgeObject* {
            auto iter = tensor_impls_.find(eid);
            if (iter == tensor_impls_.end()) {
                return nullptr;
            }
            return iter->second.get();
        });

        auto kernel = (X86OptKernel*)(info_->kernels[node_id].get());
        auto status = kernel->InferDims(&IOinfo);
        if (status != RC_SUCCESS) {
            continue;
        }
    }

    return RC_SUCCESS;
}

bool OptGraph::FuseConvAdd() {
    bool graph_changed = false;

    for (auto it = graph_->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "Add") {
            auto add_node = node;
            auto input_edge_0 = graph_->topo->GetEdgeById(node->GetInput(0));
            auto input_edge_1 = graph_->topo->GetEdgeById(node->GetInput(1));

            // check if eltwise
            auto& input_shape_0 = tensor_impls_[input_edge_0->GetId()]->GetShape();
            auto& input_shape_1 = tensor_impls_[input_edge_1->GetId()]->GetShape();
            if (input_shape_0.IsEmpty() || input_shape_1.IsEmpty()) { // input shape has not been infered
                continue;
            }
            if (input_shape_0.GetDimCount() != input_shape_1.GetDimCount()) {
                continue;
            }
            bool same_dim = true;
            for (uint32_t i = 0; i < min(input_shape_0.GetRealDimCount(), input_shape_1.GetRealDimCount()); i++) {
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
            if (!conv_node && input_edge_0->GetProducer() != INVALID_NODEID && input_edge_0->CalcConsumerCount() == 1 &&
                !IsGraphOutput(graph_, input_edge_0->GetId())) {
                auto predecessor_node_0 = graph_->topo->GetNodeById(input_edge_0->GetProducer());
                if (predecessor_node_0->GetType().domain == "" && predecessor_node_0->GetType().name == "Conv") {
                    auto conv_op = (ConvOp*)info_->kernels[predecessor_node_0->GetId()].get();
                    if (conv_op->SetFuseSum()) {
                        conv_node = predecessor_node_0;
                        src_sum_edge = input_edge_1;
                    }
                }
            }

            if (!conv_node && input_edge_1->GetProducer() != INVALID_NODEID && input_edge_1->CalcConsumerCount() == 1 &&
                !IsGraphOutput(graph_, input_edge_1->GetId())) {
                auto predecessor_node_1 = graph_->topo->GetNodeById(input_edge_1->GetProducer());
                if (predecessor_node_1->GetType().domain == "" && predecessor_node_1->GetType().name == "Conv") {
                    auto conv_op = (ConvOp*)info_->kernels[predecessor_node_1->GetId()].get();
                    if (conv_op->SetFuseSum()) {
                        conv_node = predecessor_node_1;
                        src_sum_edge = input_edge_0;
                    }
                }
            }

            if (!conv_node) {
                continue;
            }
            auto conv_output_edge_id = conv_node->GetOutput(0);
            auto add_output_edge = graph_->topo->GetEdgeById(add_node->GetOutput(0));

            conv_node->AddInput(src_sum_edge->GetId()); // add src_sum_edge as input[-1] of conv_node
            src_sum_edge->AddConsumer(conv_node->GetId());
            src_sum_edge->DelConsumer(add_node->GetId());
            conv_node->ReplaceOutput(conv_output_edge_id, add_output_edge->GetId());
            add_output_edge->SetProducer(conv_node->GetId());

            // LOG(INFO) << "fuse add " << add_node->GetName() << ".";
            info_->kernels.erase(add_node->GetId());
            graph_->topo->DelNodeById(add_node->GetId());
            graph_->topo->DelEdgeById(conv_output_edge_id);

            graph_changed = true;
        }
    }

    return graph_changed;
}

bool OptGraph::FuseChannelShuffle(const OptKernelOptions& options) {
    bool graph_changed = false;

    for (auto it = graph_->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "Reshape") {
            // find 1st Reshape node
            auto reshape1_node = node;
            auto reshape1_node_id = reshape1_node->GetId();
            auto reshape1_output_edge_id = reshape1_node->GetOutput(0);
            auto reshape1_output_edge = graph_->topo->GetEdgeById(reshape1_output_edge_id);

            if (reshape1_output_edge->CalcConsumerCount() != 1) {
                continue;
            }
            if (IsGraphOutput(graph_, reshape1_output_edge_id)) {
                continue;
            }

            // find transpose node
            auto successor_node_id = reshape1_output_edge->CreateConsumerIter().Get();
            auto successor_node = graph_->topo->GetNodeById(successor_node_id);
            if (successor_node->GetType().domain != "" || successor_node->GetType().name != "Transpose") {
                continue;
            }
            auto trans_node_id = successor_node_id;
            auto trans_node = successor_node;
            auto trans_output_edge_id = trans_node->GetOutput(0);
            auto trans_output_edge = graph_->topo->GetEdgeById(trans_output_edge_id);
            if (trans_output_edge->CalcConsumerCount() != 1) {
                continue;
            }
            if (IsGraphOutput(graph_, trans_output_edge_id)) {
                continue;
            }

            // find 2nd reshape node
            successor_node_id = trans_output_edge->CreateConsumerIter().Get();
            successor_node = graph_->topo->GetNodeById(successor_node_id);
            if (successor_node->GetType().domain != "" && successor_node->GetType().name != "Reshape") {
                continue;
            }
            auto reshape2_node = successor_node;
            auto reshape2_node_id = reshape2_node->GetId();
            auto reshape2_output_edge_id = reshape2_node->GetOutput(0);
            auto reshape2_output_edge = graph_->topo->GetEdgeById(reshape2_output_edge_id);
            if (IsGraphOutput(graph_, reshape2_output_edge_id)) {
                continue;
            }

            // check reshape input[1] kind
            auto shape1_edge_id = reshape1_node->GetInput(1);
            auto shape2_edge_id = reshape2_node->GetInput(1);
            auto shape1_edge = graph_->topo->GetEdgeById(shape1_edge_id);
            auto shape2_edge = graph_->topo->GetEdgeById(shape2_edge_id);
            if (graph_->data->constants.find(shape1_edge_id) == graph_->data->constants.end() ||
                graph_->data->constants.find(shape2_edge_id) == graph_->data->constants.end()) {
                continue;
            }

            // reshape size check
            auto& reshape1_output_shape = tensor_impls_[reshape1_output_edge_id]->GetShape();
            auto& reshape2_output_shape = tensor_impls_[reshape2_output_edge_id]->GetShape();

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
            auto& attrs = graph_->data->attrs;
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

            // add ChannelShuffle node into graph
            // base_node -> base_edge -> replace1_node -> replace1_edge -> trans_node -> trans_edge -> replace2_node ->
            // replace2_edge base_node -> base_edge -> ChannelShufflenode -> replace2_edge
            auto base_edge_id = reshape1_node->GetInput(0);
            auto base_edge = graph_->topo->GetEdgeById(base_edge_id);

            std::string channel_shuffle_node_name = "ChannelShuffle_" + reshape1_node->GetName() + "_" +
                trans_node->GetName() + "_" + reshape2_node->GetName();
            auto node_ret_pair = graph_->topo->AddNode(channel_shuffle_node_name);
            if (!node_ret_pair.second) {
                LOG(ERROR) << "node[" << channel_shuffle_node_name << "] already exists.";
                continue;
            }
            ir::Node* channel_shuffle_node = node_ret_pair.first;
            channel_shuffle_node->SetType(ir::Node::Type("ppl", "ChannelShuffle"));

            channel_shuffle_node->AddInput(base_edge_id);
            channel_shuffle_node->AddOutput(reshape2_output_edge_id);

            base_edge->DelConsumer(reshape1_node_id);
            base_edge->AddConsumer(channel_shuffle_node->GetId());

            reshape2_output_edge->SetProducer(channel_shuffle_node->GetId());

            auto type = channel_shuffle_node->GetType();
            auto creator = OptKernelCreatorManager::Instance()->Find(type.domain, type.name);
            if (!creator) {
                LOG(ERROR) << "cannot find creator for X86OptKernel[" << channel_shuffle_node->GetName() << "] type["
                           << type.domain << ":" << type.name << "]";
                continue;
            }

            auto opt_kernel = unique_ptr<X86OptKernel>(creator(channel_shuffle_node));
            if (!opt_kernel) {
                LOG(ERROR) << "create X86OptKernel failed: oom";
                continue;
            }

            auto param_ref = options.graph_data->attrs.find(opt_kernel->GetNode()->GetId());
            if (param_ref == options.graph_data->attrs.end()) {
                options.graph_data->attrs[opt_kernel->GetNode()->GetId()] = make_shared<ppl::nn::common::ChannelShuffleParam>();
            }
            else {
                LOG(ERROR) << "Node " << opt_kernel->GetNode()->GetName() << "param exist.";
                continue;
            }

            auto status = opt_kernel->Init(options);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "Init for kernel[" << opt_kernel->GetNode()->GetName()
                           << "] failed: " << GetRetCodeStr(status);
                continue;
            }
            opt_kernel->SetOutputDataFormat(0, tensor_impls_[base_edge_id]->GetShape().GetDataFormat());
            info_->kernels.emplace(channel_shuffle_node->GetId(), std::move(opt_kernel));

            // get shuffle group size
            auto channelshuffle_kernel =
                static_cast<ChannelShuffleOp*>(info_->kernels[channel_shuffle_node->GetId()].get());
            int32_t group = reshape1_output_shape.GetDim(1);
            channelshuffle_kernel->SetGroup(group);

            shape1_edge->DelConsumer(reshape1_node_id);
            shape2_edge->DelConsumer(reshape2_node_id);

            if (shape1_edge->CalcConsumerCount() == 0 && !IsGraphOutput(graph_, shape1_edge->GetId())) {
                graph_->data->constants.erase(shape1_edge_id);
                graph_->topo->DelEdgeById(shape1_edge_id);
            }
            if (shape2_edge->CalcConsumerCount() == 0 && !IsGraphOutput(graph_, shape2_edge->GetId())) {
                graph_->data->constants.erase(shape2_edge_id);
                graph_->topo->DelEdgeById(shape2_edge_id);
            }

            info_->kernels.erase(reshape1_node_id);
            info_->kernels.erase(trans_node_id);
            info_->kernels.erase(reshape2_node_id);

            graph_->topo->DelEdgeById(reshape1_output_edge_id);
            graph_->topo->DelEdgeById(trans_output_edge_id);

            graph_->topo->DelNodeById(reshape1_node_id);
            graph_->topo->DelNodeById(trans_node_id);
            graph_->topo->DelNodeById(reshape2_node_id);

            graph_changed = true;
        }
    }

    return graph_changed;
}

bool OptGraph::FuseBNReLU() {
    bool graph_changed = false;

    for (auto it = graph_->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "BatchNormalization") {
            auto bn_node = node;
            if (bn_node->GetOutputCount() > 1) { // training mode bn
                continue;
            }
            if (bn_node->GetInputCount() != 5) {
                continue;
            }
            auto bn_output_edge = graph_->topo->GetEdgeById(bn_node->GetOutput(0));
            if (!bn_output_edge || bn_output_edge->CalcConsumerCount() != 1 ||
                IsGraphOutput(graph_, bn_output_edge->GetId())) {
                continue;
            }

            auto successor_node = graph_->topo->GetNodeById(bn_output_edge->CreateConsumerIter().Get());
            if (!successor_node) {
                continue;
            }
            if (successor_node->GetType().domain != "" || successor_node->GetType().name != "Relu") {
                continue;
            }
            auto relu_node = successor_node;
            auto relu_output_edge = graph_->topo->GetEdgeById(relu_node->GetOutput(0));

            auto bn_kernel_it = info_->kernels.find(bn_node->GetId());
            if (bn_kernel_it == info_->kernels.end()) {
                continue;
            }
            auto bn_kernel = (BatchNormalizationOp*)bn_kernel_it->second.get();

            // bn_node -> bn_output_edge -> relu_node -> relu_output_edge
            bn_kernel->SetFuseReLU(true);
            bn_node->ReplaceOutput(bn_output_edge->GetId(), relu_output_edge->GetId());
            relu_output_edge->SetProducer(bn_node->GetId());

            // LOG(INFO) << "merge kernel " << bn_node->GetName() << " and " << relu_node->GetName() << ".";
            info_->kernels.erase(relu_node->GetId());
            graph_->topo->DelNodeById(relu_node->GetId());
            graph_->topo->DelEdgeById(bn_output_edge->GetId());

            graph_changed = true;
        }
    }

    return graph_changed;
}

bool OptGraph::FuseArithmeticReLU() {
    bool graph_changed = false;

    for (auto it = graph_->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
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
            auto arithmetic_output_edge = graph_->topo->GetEdgeById(arithmetic_node->GetOutput(0));
            if (!arithmetic_output_edge || arithmetic_output_edge->CalcConsumerCount() != 1 ||
                IsGraphOutput(graph_, arithmetic_output_edge->GetId())) {
                continue;
            }
            auto arithmetic_output_edge_shape = tensor_impls_[arithmetic_output_edge->GetId()]->GetShape();

            // Only Support FP32
            if (arithmetic_output_edge_shape.GetDataType() != DATATYPE_FLOAT32) {
                continue;
            }

            auto successor_node = graph_->topo->GetNodeById(arithmetic_output_edge->CreateConsumerIter().Get());
            if (!successor_node) {
                continue;
            }
            if (successor_node->GetType().domain != "" || successor_node->GetType().name != "Relu") {
                continue;
            }
            auto relu_node = successor_node;
            auto relu_output_edge = graph_->topo->GetEdgeById(relu_node->GetOutput(0));

            auto arithmetic_kernel_it = info_->kernels.find(arithmetic_node->GetId());
            if (arithmetic_kernel_it == info_->kernels.end()) {
                continue;
            }

            // arithmetic_node -> arithmetic_output_edge -> relu_node -> relu_output_edge
            if (at == ADD) {
                auto arithmetic_kernel = (AddOp*)arithmetic_kernel_it->second.get();
                arithmetic_kernel->SetFuseReLU(true);
            } else if (at == SUB) {
                auto arithmetic_kernel = (SubOp*)arithmetic_kernel_it->second.get();
                arithmetic_kernel->SetFuseReLU(true);
            } else if (at == MUL) {
                auto arithmetic_kernel = (MulOp*)arithmetic_kernel_it->second.get();
                arithmetic_kernel->SetFuseReLU(true);
            } else if (at == DIV) {
                auto arithmetic_kernel = (DivOp*)arithmetic_kernel_it->second.get();
                arithmetic_kernel->SetFuseReLU(true);
            } else {
                continue;
            }

            arithmetic_node->ReplaceOutput(arithmetic_output_edge->GetId(), relu_output_edge->GetId());
            relu_output_edge->SetProducer(arithmetic_node->GetId());

            info_->kernels.erase(relu_node->GetId());
            graph_->topo->DelNodeById(relu_node->GetId());
            graph_->topo->DelEdgeById(arithmetic_output_edge->GetId());

            graph_changed = true;
        }
    }
    return graph_changed;
}

bool OptGraph::FuseFcActivation() {
    bool graph_changed = false;

    for (auto it = graph_->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "Gemm") {
            auto fc_node = node;
            auto fc_output_edge_id = fc_node->GetOutput(0);
            auto fc_output_edge = graph_->topo->GetEdgeById(fc_output_edge_id);
            if (fc_output_edge->CalcConsumerCount() != 1) {
                continue;
            }
            if (IsGraphOutput(graph_, fc_output_edge_id)) {
                continue;
            }

            auto successor_node_id = fc_output_edge->CreateConsumerIter().Get();
            auto successor_node = graph_->topo->GetNodeById(successor_node_id);
            if (successor_node->GetType().domain != "") {
                continue;
            }

            auto fc_kernel = static_cast<GemmOp*>(info_->kernels[fc_node->GetId()].get());
            if (successor_node->GetType().name == "Relu") {
                if (!fc_kernel->SetFuseReLU()) { // set fuse flag to fc_op
                    continue;
                }
            } else {
                continue;
            }

            auto activation_node = successor_node;
            auto activation_node_id = activation_node->GetId();
            auto activation_output_edge_id = activation_node->GetOutput(0);
            auto activation_output_edge = graph_->topo->GetEdgeById(activation_output_edge_id);
            // fc_node -> fc_output_edge -> activation_node -> activation_output_edge
            // fc_node                                      -> activation_output_edge
            fc_node->ReplaceOutput(fc_output_edge_id, activation_output_edge_id);
            activation_output_edge->SetProducer(fc_node->GetId());

            // LOG(INFO) << "merge kernel " << activation_node->GetName() << " into kernel " <<
            // fc_node->GetName() << ".";
            info_->kernels.erase(activation_node_id);
            tensor_impls_.erase(fc_output_edge_id);
            graph_->topo->DelNodeById(activation_node_id);
            graph_->topo->DelEdgeById(fc_output_edge_id);

            graph_changed = true;
        }
    }

    return graph_changed;
}

ppl::common::RetCode OptGraph::CreateX86OptKernel(const OptKernelOptions& options, const ir::Node* node,
                                                  X86OptKernel** kernel) {
    auto& type = node->GetType();

    auto creator = OptKernelCreatorManager::Instance()->Find(type.domain, type.name);
    if (!creator) {
        LOG(ERROR) << "cannot find creator for X86OptKernel[" << node->GetName() << "] type[" << type.domain << ":"
                   << type.name << "]";
        return RC_NOT_FOUND;
    }

    auto opt_kernel = unique_ptr<X86OptKernel>(creator(node));
    if (!opt_kernel) {
        LOG(ERROR) << "create X86OptKernel failed: oom";
        return RC_OUT_OF_MEMORY;
    }

    auto status = opt_kernel->Init(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Init for kernel[" << node->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }
    *kernel = opt_kernel.get();
    info_->kernels.emplace(node->GetId(), std::move(opt_kernel));

    return RC_SUCCESS;
}

bool OptGraph::FuseSwish(const OptKernelOptions& options) {
    bool graph_changed = false;

    for (auto it = graph_->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain == "" && node->GetType().name == "Sigmoid") {
            auto sigmoid_node = node;
            auto sigmoid_output_edge = graph_->topo->GetEdgeById(sigmoid_node->GetOutput(0));
            if (sigmoid_output_edge->CalcConsumerCount() != 1 || IsGraphOutput(graph_, sigmoid_output_edge->GetId())) {
                continue;
            }
            auto sigmoid_input_edge = graph_->topo->GetEdgeById(sigmoid_node->GetInput(0));

            auto successor_node = graph_->topo->GetNodeById(sigmoid_output_edge->CreateConsumerIter().Get());
            if (!successor_node || successor_node->GetType().domain != "" || successor_node->GetType().name != "Mul") {
                continue;
            }
            auto last_mul_node = successor_node;

            auto last_mul_input_edge0 = graph_->topo->GetEdgeById(last_mul_node->GetInput(0));
            auto last_mul_input_edge1 = graph_->topo->GetEdgeById(last_mul_node->GetInput(1));
            if (!last_mul_input_edge0 || !last_mul_input_edge1) {
                continue;
            }
            auto last_mul_output_edge = graph_->topo->GetEdgeById(last_mul_node->GetOutput(0));

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
                const ir::Node::Type type("ppl", "Swish");

                // add node to graph topo
                auto node_ret_pair = graph_->topo->AddNode(swish_node_name);
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
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Create OptKernel [" << swish_node_name << "] failed: " << GetRetCodeStr(status);
                    graph_->topo->DelNodeById(node->GetId());
                    continue;
                }

                ((SwishOp*)swish_opt_kernel)->SetBeta(1.0f); // default to 1.0
                swish_opt_kernel->SetOutputDataFormat(
                    0, tensor_impls_[last_mul_output_edge->GetId()].get()->GetShape().GetDataFormat());

                // change graph topo
                sigmoid_input_edge->DelConsumer(sigmoid_node->GetId());
                sigmoid_input_edge->DelConsumer(last_mul_node->GetId());
                sigmoid_input_edge->AddConsumer(swish_node->GetId());
                last_mul_output_edge->SetProducer(swish_node->GetId());

                // delete unused node & edge
                info_->kernels.erase(sigmoid_node->GetId());
                info_->kernels.erase(last_mul_node->GetId());
                tensor_impls_.erase(sigmoid_output_edge->GetId());

                // LOG(INFO) << "successfully merged node " << sigmoid_node->GetName() << " and "
                //            << last_mul_node->GetName() << " into node " << swish_node->GetName() << ".";
                graph_->topo->DelEdgeById(sigmoid_output_edge->GetId());
                graph_->topo->DelNodeById(sigmoid_node->GetId());
                graph_->topo->DelNodeById(last_mul_node->GetId());

                graph_changed = true;
            }

            // TODO: fuse swish with beta(another mul op)
        }
    }

    return graph_changed;
}

RetCode OptGraph::DoOptimize(X86Device* device) {
    OptKernelOptions options;
    options.resource = resource_;
    options.graph_data = graph_->data.get();
    options.device = device;

    for (auto it = info_->kernels.begin(); it != info_->kernels.end(); ++it) {
        auto kernel = (X86OptKernel*)(it->second.get());
        auto status = kernel->Init(options);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Init for kernel[" << kernel->GetNode()->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    for (auto it = tensor_impls_.begin(); it != tensor_impls_.end(); ++it) {
        auto edge_id = it->first;
        if (graph_->data->constants.find(edge_id) != graph_->data->constants.end()) {
            auto tensor = it->second.get();
            tensor->SetDevice(device);
            tensor->ReallocBuffer();
            memcpy(tensor->GetBufferPtr<void>(), graph_->data->constants[edge_id].data.data(),
                   tensor->GetShape().GetBytesExcludingPadding());
        }
    }

    RetCode status = RC_SUCCESS;

    status = TryToInferType(device);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "TryToInferType failed: " << GetRetCodeStr(status);
        return status;
    }

    status = TryToInferDims(device);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "TryToInferDims failed: " << GetRetCodeStr(status);
        return status;
    }

    FuseChannelShuffle(options);

    status = LayoutOptimize(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "LayoutOptimize failed: " << GetRetCodeStr(status);
        return status;
    }

    while (FuseConvActivation() || FuseConvAdd() || FuseBNReLU() || FuseArithmeticReLU() || FuseFcActivation() ||
           FuseSwish(options))
        ;

#ifdef SHOW_GRAPH_VIS
    std::string vis = utils::ToGraphviz(graph_->topo.get());
    std::ofstream out_file("./graph.dot");
    if (out_file.is_open()) {
        out_file << vis;
    }
#endif

    for (auto it = tensor_impls_.begin(); it != tensor_impls_.end(); ++it) {
        auto edge_id = it->first;
        if (graph_->data->constants.find(edge_id) != graph_->data->constants.end()) {
            auto tensor = it->second.get();
            tensor->FreeBuffer();
        }
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::x86
