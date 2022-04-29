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

#include "ppl/nn/engines/riscv/optimizer/rules/utils.h"
#include "ppl/nn/engines/riscv/optimizer/opt_rule_manager.h"

namespace ppl { namespace nn { namespace riscv {

#define REORDER_INPUT 0
#define REORDER_OUTPUT 1
#define REORDER_EXTRA_INPUT 2

static ppl::common::RetCode AddReorderOp(const OptKernelOptions& options, const edgeid_t& edge_id,
                                         const nodeid_t& node_id, const int32_t& reorder_type,
                                         const ppl::common::dataformat_t& reorder_src_format,
                                         const ppl::common::dataformat_t& reorder_dst_format,
                                         const ppl::common::datatype_t& reorder_src_type,
                                         const ppl::common::datatype_t& reorder_dst_type) {
    auto graph_topo = options.graph_topo;
    auto info = options.info;
    auto& tensors = *options.tensors;

    auto edge = graph_topo->GetEdge(edge_id);
    auto node = graph_topo->GetNode(node_id);

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
    reorder_node->SetType(ir::Node::Type("pmx", "Reorder", 1));

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

    auto type = reorder_node->GetType();
    auto creator = OptKernelCreatorManager::GetInstance()->Find(type.domain, type.name, type.version);
    if (!creator) {
        LOG(ERROR) << "cannot find creator for RiscvOptKernel[" << reorder_node->GetName() << "] type[" << type.domain
                   << ":" << type.name << "]";
        return ppl::common::RC_NOT_FOUND;
    }

    auto opt_kernel = std::unique_ptr<RiscvOptKernel>((*creator)(reorder_node));
    if (!opt_kernel) {
        LOG(ERROR) << "create RiscvOptKernel failed: oom";
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    auto status = opt_kernel->Init(options);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "Init for kernel[" << opt_kernel->GetNode()->GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }

    opt_kernel->SetOutputDataFormat(0, reorder_dst_format);
    opt_kernel->SetOutputDataType(0, reorder_dst_type);

    info->kernels.emplace(reorder_node->GetId(), std::move(opt_kernel));

    TensorImpl* tensor = new TensorImpl(reorder_edge, TENSORTYPE_NORMAL);

    tensor->GetShape()->SetDataFormat(reorder_dst_format);
    tensor->GetShape()->SetDataType(reorder_dst_type);

    tensors.emplace(reorder_edge->GetId(), std::unique_ptr<TensorImpl>(tensor));

    LOG(DEBUG) << "successfully add reorder op " << reorder_node_name << " to reorder "
               << ppl::common::GetDataFormatStr(reorder_src_format) << " "
               << ppl::common::GetDataTypeStr(reorder_src_type) << " to "
               << ppl::common::GetDataFormatStr(reorder_dst_format) << " "
               << ppl::common::GetDataTypeStr(reorder_dst_type) << ".";
    return ppl::common::RC_SUCCESS;
}

bool LayoutOptimize(const OptKernelOptions& options) {
    auto graph_topo = options.graph_topo;
    auto info = options.info;
    auto& tensors = *options.tensors;

    std::vector<nodeid_t> sorted_nodes;
    graph_topo->TopologicalSort([&sorted_nodes](nodeid_t nid) -> void {
        sorted_nodes.push_back(nid);
    });

    for (auto node_id : sorted_nodes) {
        if (info->kernels.find(node_id) == info->kernels.end()) {
            LOG(ERROR) << "cannot find node_id " << node_id << " in RuntimePartitionInfo.";
            return false;
        }
        auto kernel = (RiscvOptKernel*)info->kernels[node_id].get();
        auto node = kernel->GetNode();

        InputOutputInfo IOinfo;
        IOinfo.SetNode(node);
        IOinfo.SetAcquireFunc([&tensors](edgeid_t eid, uint32_t) -> EdgeObject* {
            auto iter = tensors.find(eid);
            if (iter == tensors.end()) {
                return nullptr;
            }
            return iter->second.get();
        });

        {
            auto status = kernel->SelectAlgorithm(IOinfo, options);
            if (status != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "kernel[" << node->GetName()
                           << "] SelectAlgorithm failed: " << ppl::common::GetRetCodeStr(status);
                return false;
            }
        }

        auto forward_precision = options.engine_options->forward_precision;
        std::vector<ppl::common::dataformat_t> selected_input_formats(node->GetInputCount(),
                                                                      ppl::common::DATAFORMAT_NDARRAY);
        std::vector<ppl::common::datatype_t> selected_input_data_types(node->GetInputCount(),
                                                                       ppl::common::DATATYPE_FLOAT32);
        std::vector<ppl::common::dataformat_t> selected_output_formats(node->GetOutputCount(),
                                                                       ppl::common::DATAFORMAT_NDARRAY);
        std::vector<ppl::common::datatype_t> selected_output_data_types(node->GetOutputCount(),
                                                                        ppl::common::DATATYPE_FLOAT32);
        {
            for (uint32_t i = 0; i < node->GetInputCount(); i++) {
                auto edge_id = node->GetInput(i);
                if (edge_id == INVALID_EDGEID) {
                    continue;
                }
                selected_input_formats[i] = tensors[edge_id]->GetShape()->GetDataFormat();
                selected_input_data_types[i] = tensors[edge_id]->GetShape()->GetDataType();
            }

            for (uint32_t i = 0; i < node->GetOutputCount(); i++) {
                auto edge_id = node->GetOutput(i);
                if (edge_id == INVALID_EDGEID) {
                    continue;
                }
                selected_output_formats[i] = tensors[edge_id]->GetShape()->GetDataFormat();
                selected_output_data_types[i] = tensors[edge_id]->GetShape()->GetDataType();
            }

            auto status = kernel->SelectInputOutput(IOinfo, forward_precision,
                &selected_input_formats, &selected_output_formats, &selected_input_data_types, &selected_output_data_types);
            if (status != ppl::common::RC_SUCCESS) {
                return false;
            }
        }

        for (uint32_t i = 0; i < node->GetInputCount(); i++) {
            auto edge_id = node->GetInput(i);
            if (edge_id == INVALID_EDGEID) {
                continue;
            }
            auto input_format = tensors[edge_id]->GetShape()->GetDataFormat();
            auto input_data_type = tensors[edge_id]->GetShape()->GetDataType();
            auto selected_input_format = selected_input_formats[i];
            auto selected_input_data_type = selected_input_data_types[i];
            if (input_format != selected_input_format || input_data_type != selected_input_data_type) {
                auto status = AddReorderOp(options, edge_id, node_id, REORDER_INPUT, input_format,
                                           selected_input_format, input_data_type, selected_input_data_type);
                if (status != ppl::common::RC_SUCCESS) {
                    LOG(ERROR) << "add reorder op failed.";
                    return false;
                }
            }
        }

        // extra input(used by if/loop op) force to be ndarray
        for (uint32_t i = 0; i < node->GetExtraInputCount(); i++) {
            auto edge_id = node->GetExtraInput(i);
            auto extra_input_format = tensors[edge_id]->GetShape()->GetDataFormat();
            auto extra_input_data_type = tensors[edge_id]->GetShape()->GetDataType();
            if (extra_input_format != ppl::common::DATAFORMAT_NDARRAY) {
                auto status =
                    AddReorderOp(options, edge_id, node_id, REORDER_EXTRA_INPUT, extra_input_format,
                                 ppl::common::DATAFORMAT_NDARRAY, extra_input_data_type, extra_input_data_type);
                if (status != ppl::common::RC_SUCCESS) {
                    LOG(ERROR) << "add reorder op failed.";
                    return false;
                }
            }
        }

        for (uint32_t i = 0; i < node->GetOutputCount(); i++) {
            auto edge_id = node->GetOutput(i);
            auto output_format = ppl::common::DATAFORMAT_NDARRAY;
            auto output_type = selected_output_data_types[i];

            auto selected_output_format = selected_output_formats[i];
            auto selected_output_data_type = selected_output_data_types[i];

            tensors[edge_id]->GetShape()->SetDataFormat(selected_output_format);
            tensors[edge_id]->GetShape()->SetDataType(selected_output_data_type);
            kernel->SetOutputDataFormat(i, selected_output_format);
            kernel->SetOutputDataType(i, selected_output_data_type);
            if (IsGraphOutput(graph_topo, edge_id) && selected_output_format != output_format) {
                auto status = AddReorderOp(options, edge_id, node_id, REORDER_OUTPUT, selected_output_format,
                                           output_format, selected_output_data_type, output_type);
                if (status != ppl::common::RC_SUCCESS) {
                    LOG(ERROR) << "add reorder op failed.";
                    return false;
                }
            }
        }
    }

    return true;
}

}}} // namespace ppl::nn::riscv
