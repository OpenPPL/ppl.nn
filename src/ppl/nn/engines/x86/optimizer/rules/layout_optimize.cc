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

namespace ppl { namespace nn { namespace x86 {

#define REORDER_INPUT 0
#define REORDER_OUTPUT 1
#define REORDER_EXTRA_INPUT 2

static ppl::common::RetCode AddReorderOp(
    const OptKernelOptions& options,
    const edgeid_t& edge_id,
    const nodeid_t& node_id,
    const int32_t& reorder_type,
    const ppl::common::dataformat_t& reorder_in_format,
    const ppl::common::dataformat_t& reorder_out_format)
{
    auto graph_topo = options.graph_topo;
    auto info = options.info;
    auto &tensors = *options.tensors;

    auto edge = graph_topo->GetEdgeById(edge_id);
    auto node = graph_topo->GetNodeById(node_id);

    std::string reorder_node_name = "";
    if (reorder_type == REORDER_INPUT) {
        reorder_node_name = "ReorderInput_" + edge->GetName() + "_of_" + node->GetName();
    } else if (reorder_type == REORDER_OUTPUT) {
        reorder_node_name = "ReorderOutput_" + edge->GetName() + "_of_" + node->GetName();
    } else if (reorder_type == REORDER_EXTRA_INPUT) {
        reorder_node_name = "ReorderExtraInput_" + edge->GetName() + "_of_" + node->GetName();
    }

    auto node_ret_pair = graph_topo->AddNode(reorder_node_name);
    if (!node_ret_pair.second) {
        LOG(ERROR) << "node[" << reorder_node_name << "] already exists.";
        return ppl::common::RC_EXISTS;
    }
    ir::Node* reorder_node = node_ret_pair.first; // TODO: change name for easy to understand
    reorder_node->SetType(ir::Node::Type("ppl", "Reorder", 1));

    std::string reorder_edge_name = reorder_node_name + "_edge";
    auto edge_ret_pair = graph_topo->AddEdge(reorder_edge_name);
    if (!edge_ret_pair.second) {
        LOG(ERROR) << "edge[" << reorder_edge_name << "] already exists.";
        return ppl::common::RC_EXISTS;
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

    auto& type = reorder_node->GetType();
    auto creator = OptKernelCreatorManager::Instance()->Find(type.domain, type.name, type.version);
    if (!creator) {
        LOG(ERROR) << "cannot find creator for X86OptKernel[" << reorder_node->GetName() << "] type[" << type.domain
                   << ":" << type.name << "]";
        return ppl::common::RC_NOT_FOUND;
    }

    auto opt_kernel = std::unique_ptr<X86OptKernel>(creator(reorder_node));
    if (!opt_kernel) {
        LOG(ERROR) << "create X86OptKernel failed: oom";
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    auto status = opt_kernel->Init(options);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "Init for kernel[" << opt_kernel->GetNode()->GetName() << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    opt_kernel->SetOutputDataFormat(0, reorder_out_format);
    info->kernels.emplace(reorder_node->GetId(), std::move(opt_kernel));

    TensorImpl* tensor = new TensorImpl(reorder_edge, TENSORTYPE_NORMAL);
    tensor->GetShape().SetDataFormat((reorder_type == REORDER_INPUT || reorder_type == REORDER_EXTRA_INPUT)
                                         ? reorder_out_format
                                         : reorder_in_format);
    tensors.emplace(reorder_edge->GetId(), std::unique_ptr<TensorImpl>(tensor));

    // LOG(INFO) << "successfully add reorder op " << reorder_node_name << " to reorder " <<
    // GetDataFormatStr(reorder_in_format) << " to " << GetDataFormatStr(reorder_out_format) << ".";
    return ppl::common::RC_SUCCESS;
}

static ppl::common::RetCode FuseReorderOp(const OptKernelOptions& options) {
    auto graph_topo = options.graph_topo;
    auto info = options.info;
    auto &tensors = *options.tensors;

    std::vector<nodeid_t> sorted_nodes;
    graph_topo->TopologicalSort([&sorted_nodes](nodeid_t nid) -> void {
        sorted_nodes.push_back(nid);
    });

    for (auto node_it = graph_topo->CreateNodeIter(); node_it->IsValid(); node_it->Forward()) {
        auto node = node_it->Get();
        if (node->GetType().domain == "ppl" && node->GetType().name == "Reorder") {
            auto input_edge_id = node->GetInput(0);
            auto input_edge = graph_topo->GetEdgeById(input_edge_id);

            // find reorder op group with same output_format so that can be merged together
            std::map<ppl::common::dataformat_t, std::vector<ir::Node*>> reorder_op_groups;
            for (auto it = input_edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
                auto consumer_node = graph_topo->GetNodeById(it.Get());
                if (consumer_node->GetType().domain == "ppl" && consumer_node->GetType().name == "Reorder") {
                    auto output_edge_id = consumer_node->GetOutput(0);
                    auto output_format = tensors[output_edge_id]->GetShape().GetDataFormat();

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
                auto merged_output_edge = graph_topo->GetEdgeById(merged_output_edge_id);
                for (uint32_t i = 1; i < op_group.size(); i++) {
                    auto del_node_id = op_group[i]->GetId();
                    auto del_output_edge_id = op_group[i]->GetOutput(0);
                    auto del_output_edge = graph_topo->GetEdgeById(del_output_edge_id);
                    auto consumer_node_id = del_output_edge->CreateConsumerIter().Get(); // only has one consumer
                    auto consumer_node = graph_topo->GetNodeById(consumer_node_id);

                    merged_output_edge->AddConsumer(consumer_node_id);
                    consumer_node->ReplaceInput(del_output_edge_id, merged_output_edge_id);
                    input_edge->DelConsumer(del_node_id);

                    // LOG(INFO) << "kernel " << op_group[i]->GetName() << " is merged into kernel " <<
                    // op_group[0]->GetName() << ".";
                    info->kernels.erase(del_node_id);
                    graph_topo->DelNodeById(del_node_id);
                    graph_topo->DelEdgeById(del_output_edge_id);
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

bool LayoutOptimize(const OptKernelOptions &options) {
    auto graph_topo = options.graph_topo;
    auto info = options.info;
    auto &tensors = *options.tensors;

    std::vector<nodeid_t> sorted_nodes;
    graph_topo->TopologicalSort([&sorted_nodes](nodeid_t nid) -> void {
        sorted_nodes.push_back(nid);
    });

    for (auto node_id : sorted_nodes) {
        if (info->kernels.find(node_id) == info->kernels.end()) {
            LOG(ERROR) << "cannot find node_id " << node_id << " in RuntimePartitionInfo.";
            return false;
        }
        auto kernel = (X86OptKernel*)info->kernels[node_id].get();
        auto node = kernel->GetNode();

        InputOutputInfo IOinfo;
        IOinfo.SetNode(node);
        IOinfo.SetAcquireObjectFunc([&tensors](edgeid_t eid, uint32_t, Device*) -> EdgeObject* {
            auto iter = tensors.find(eid);
            if (iter == tensors.end()) {
                return nullptr;
            }
            return iter->second.get();
        });

        auto status = kernel->SelectAlgorithm(IOinfo, options);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "kernel[" << node->GetName() << "] SelectAlgorithm failed: " << ppl::common::GetRetCodeStr(status);
            return false;
        }

        std::vector<ppl::common::dataformat_t> selected_input_formats(node->GetInputCount(), ppl::common::DATAFORMAT_NDARRAY);
        std::vector<ppl::common::dataformat_t> selected_output_formats(node->GetOutputCount(), ppl::common::DATAFORMAT_NDARRAY);

        status = kernel->SelectFormat(IOinfo, &selected_input_formats, &selected_output_formats);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "kernel[" << node->GetName() << "] SelectFormat failed: " << ppl::common::GetRetCodeStr(status);
            return false;
        }

        for (uint32_t i = 0; i < node->GetInputCount(); i++) {
            auto edge_id = node->GetInput(i);
            if (edge_id == INVALID_EDGEID) {
                continue;
            }
            auto input_format = tensors[edge_id]->GetShape().GetDataFormat();
            auto selected_input_format = selected_input_formats[i];
            if (input_format != selected_input_format) {
                status = AddReorderOp(options, edge_id, node_id, REORDER_INPUT, input_format, selected_input_format);
                if (status != ppl::common::RC_SUCCESS) {
                    LOG(ERROR) << "add reorder op failed.";
                    return false;
                }
            }
        }

        // extra input(used by if/loop op) force to be ndarray
        for (uint32_t i = 0; i < node->GetExtraInputCount(); i++) {
            auto edge_id = node->GetExtraInput(i);
            auto extra_input_format = tensors[edge_id]->GetShape().GetDataFormat();
            if (extra_input_format != ppl::common::DATAFORMAT_NDARRAY) {
                status = AddReorderOp(options, edge_id, node_id, REORDER_EXTRA_INPUT, extra_input_format,
                                      ppl::common::DATAFORMAT_NDARRAY);
                if (status != ppl::common::RC_SUCCESS) {
                    LOG(ERROR) << "add reorder op failed.";
                    return false;
                }
            }
        }

        for (uint32_t i = 0; i < node->GetOutputCount(); i++) {
            auto edge_id = node->GetOutput(i);
            auto selected_output_format = selected_output_formats[i];
            tensors[edge_id]->GetShape().SetDataFormat(selected_output_format);
            kernel->SetOutputDataFormat(i, selected_output_format);
        }
    }

    auto status = FuseReorderOp(options);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "FuseReorderOp failed: " << ppl::common::GetRetCodeStr(status);
        return false;
    }

    return true;
}

}}} // namespace ppl::nn::x86

