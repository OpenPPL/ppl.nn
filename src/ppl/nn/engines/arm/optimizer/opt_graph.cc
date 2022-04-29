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

#include "ppl/nn/engines/arm/optimizer/opt_graph.h"

#include <string.h>

#include "ppl/nn/engines/arm/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/arm/optimizer/opt_rule_manager.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/conv_op.h"
#include "ppl/nn/engines/utils.h"

//#define SHOW_GRAPH_VIS
#ifdef SHOW_GRAPH_VIS
#include "ppl/nn/auxtools/to_graphviz.h"
#include <fstream>
#endif

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

RetCode OptGraph::InitKernels(const ir::Graph* graph) {
    auto topo = graph->topo.get();
    for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        auto& type = node->GetType();
        auto creator = OptKernelCreatorManager::GetInstance()->Find(type.domain, type.name, type.version);
        if (!creator) {
            LOG(ERROR) << "cannot find creator for ArmOptKernel[" << node->GetName() << "] type[" << type.domain << ":"
                       << type.name << "]";
            return RC_NOT_FOUND;
        }

        auto opt_kernel = unique_ptr<ArmOptKernel>((*creator)(node));
        if (!opt_kernel) {
            LOG(ERROR) << "create ArmOptKernel failed: oom";
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
            utils::IrShape2TensorShape(shapes[edge_id], tensor->GetShape());
        } else {
            tensor->GetShape()->SetDataFormat(DATAFORMAT_NDARRAY);
        }
        tensor_impls_.emplace(edge_id, unique_ptr<TensorImpl>(tensor));
    }
    return RC_SUCCESS;
}

RetCode OptGraph::Init(ir::Graph* graph, RuntimePartitionInfo* info, ArmEngineOptions* options) {
    graph_ = graph;
    info_ = info;
    options_ = options;

    OptKernelOptions opt_kernel_options;
    opt_kernel_options.graph_data = graph_->data.get(); // only a part of info can be used on Init stage
    opt_kernel_options.graph_topo = graph_->topo.get();
    opt_kernel_options.info = info_;
    opt_kernel_options.engine_options = options_;

    // do before init optimize
    const auto opt_rule_manager = OptRuleManager::Instance();
    const auto max_opt_level = opt_rule_manager->GetMaxOptLevel(options_->graph_optimization_level);
    auto status = opt_rule_manager->ApplyRules(opt_kernel_options, max_opt_level, "BeforeInitOptimize", "");
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Run BeforeInitOptimize failed: " << GetRetCodeStr(status);
        return status;
    }

    status = InitKernels(graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init kernels failed: " << GetRetCodeStr(status);
        return status;
    }

    status = InitTensorImpls();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init tensor impls failed: " << GetRetCodeStr(status);
        return status;
    }

    acquire_tensor_func_ = [this](edgeid_t eid, uint32_t) -> EdgeObject* {
        auto it = tensor_impls_.find(eid);
        if (it == tensor_impls_.end()) {
            return nullptr;
        }
        return it->second.get();
    };

    return RC_SUCCESS;
}

#define REORDER_INPUT 0
#define REORDER_OUTPUT 1
#define REORDER_EXTRA_INPUT 2

RetCode OptGraph::AddReorderOp(const OptKernelOptions& options, const edgeid_t& edge_id, const nodeid_t& node_id,
                               const int32_t& reorder_type, const ppl::common::dataformat_t& reorder_in_format,
                               const ppl::common::dataformat_t& reorder_out_format,
                               const ppl::common::datatype_t& reorder_in_type,
                               const ppl::common::datatype_t& reorder_out_type) {
    auto edge = graph_->topo->GetEdge(edge_id);
    auto node = graph_->topo->GetNode(node_id);

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
    ir::Node* reorder_node = node_ret_pair.first;
    reorder_node->SetType(ir::Node::Type("pmx", "Reorder", 1));

    std::string reorder_edge_name = reorder_node_name + "_edge";
    auto edge_ret_pair = graph_->topo->AddEdge(reorder_edge_name);
    if (!edge_ret_pair.second) {
        LOG(ERROR) << "edge[" << reorder_edge_name << "] already exists.";
        return RC_EXISTS;
    }
    ir::Edge* reorder_edge = edge_ret_pair.first;

    if (reorder_type == REORDER_INPUT || reorder_type == REORDER_EXTRA_INPUT) {
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
    } else if (reorder_type == REORDER_OUTPUT) {
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
        LOG(ERROR) << "cannot find creator for ArmOptKernel[" << reorder_node->GetName() << "] type[" << type.domain
                   << ":" << type.name << "]";
        return RC_NOT_FOUND;
    }

    auto opt_kernel = unique_ptr<ArmOptKernel>((*creator)(reorder_node));
    if (!opt_kernel) {
        LOG(ERROR) << "create ArmOptKernel failed: oom";
        return RC_OUT_OF_MEMORY;
    }

    auto status = opt_kernel->Init(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Init for kernel[" << opt_kernel->GetNode()->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }
    opt_kernel->SetOutputDataFormat(0, reorder_out_format);
    opt_kernel->SetOutputDataType(0, reorder_out_type);

    info_->kernels.emplace(reorder_node->GetId(), std::move(opt_kernel));

    TensorImpl* tensor = new TensorImpl(reorder_edge, TENSORTYPE_NORMAL);
    tensor->GetShape()->SetDataFormat((reorder_type == REORDER_INPUT || reorder_type == REORDER_EXTRA_INPUT)
                                          ? reorder_out_format
                                          : reorder_in_format);
    tensor->GetShape()->SetDataType(
        (reorder_type == REORDER_INPUT || reorder_type == REORDER_EXTRA_INPUT) ? reorder_out_type : reorder_in_type);

    tensor_impls_.emplace(reorder_edge->GetId(), unique_ptr<TensorImpl>(tensor));

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    LOG(INFO) << "Successfully added reorder op " << reorder_node_name << ". [" << GetDataFormatStr(reorder_in_format)
              << ", " << GetDataTypeStr(reorder_in_type) << "] --> [" << GetDataFormatStr(reorder_out_format) << ", "
              << GetDataTypeStr(reorder_out_type) << "]";
#endif
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

inline bool IsGraphInput(const ir::Graph* graph, edgeid_t edge_id) {
    for (uint32_t i = 0; i < graph->topo->GetInputCount(); i++) {
        if (graph->topo->GetInput(i) == edge_id) {
            return true;
        }
    }
    return false;
}

RetCode OptGraph::StitchGraph(const OptKernelOptions& options) {
    vector<nodeid_t> sorted_nodes;
    graph_->topo->TopologicalSort([&sorted_nodes](nodeid_t nid) -> void {
        sorted_nodes.push_back(nid);
    });

    for (auto node_id : sorted_nodes) {
        if (info_->kernels.find(node_id) == info_->kernels.end()) {
            LOG(ERROR) << "Cannot find node_id " << node_id << " in RuntimePartitionInfo.";
            return RC_NOT_FOUND;
        }
        auto kernel = (ArmOptKernel*)info_->kernels[node_id].get();
        auto node = kernel->GetNode();

        InputOutputInfo IOinfo;
        IOinfo.SetNode(node);
        IOinfo.SetAcquireFunc(acquire_tensor_func_);

        auto status = kernel->SelectAlgorithm(IOinfo, options);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Kernel[" << node->GetName() << "] SelectAlgorithm failed: " << GetRetCodeStr(status);
            return status;
        }

        vector<datatype_t> selected_input_types(node->GetInputCount(), ppl::common::DATATYPE_UNKNOWN);
        vector<datatype_t> selected_output_types(node->GetOutputCount(), ppl::common::DATATYPE_UNKNOWN);

        status =
            kernel->SelectDataType(IOinfo, &selected_input_types, &selected_output_types, options_->forward_precision);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Kernel[" << node->GetName() << "] SelectDataType failed: " << GetRetCodeStr(status);
            return status;
        }

        for (uint32_t i = 0; i < selected_input_types.size(); i++) {
            if (selected_input_types[i] == ppl::common::DATATYPE_UNKNOWN) {
                LOG(ERROR) << "Kernel[" << node->GetName() << "] not selected input data type for input " << i;
            }
        }
        for (uint32_t i = 0; i < selected_output_types.size(); i++) {
            if (selected_output_types[i] == ppl::common::DATATYPE_UNKNOWN) {
                LOG(ERROR) << "Kernel[" << node->GetName() << "] not selected output data type for output " << i;
            }
        }

        vector<dataformat_t> selected_input_formats(node->GetInputCount(), DATAFORMAT_NDARRAY);
        vector<dataformat_t> selected_output_formats(node->GetOutputCount(), DATAFORMAT_NDARRAY);

        status = kernel->SelectFormat(IOinfo, &selected_input_formats, &selected_output_formats);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Kernel[" << node->GetName() << "] SelectFormat failed: " << GetRetCodeStr(status);
            return status;
        }
        for (uint32_t i = 0; i < node->GetInputCount(); i++) {
            auto edge_id = node->GetInput(i);
            if (edge_id == INVALID_EDGEID) {
                continue;
            }
            auto input_format = tensor_impls_[edge_id]->GetShape()->GetDataFormat();
            auto input_type = tensor_impls_[edge_id]->GetShape()->GetDataType();
            auto selected_input_format = selected_input_formats[i];
            auto selected_input_type = selected_input_types[i];
            // default input is always fp32 and ndarray
            if (IsGraphInput(graph_, edge_id)) {
                input_type = DATATYPE_FLOAT32;
                input_format = DATAFORMAT_NDARRAY;
            }
            if (input_format != selected_input_format || input_type != selected_input_type) {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
                LOG(INFO) << "In Stitching: " << node->GetName() << "\'s input[" << i << "] requires ["
                          << GetDataFormatStr(input_format) << ", " << GetDataTypeStr(input_type) << "] --> ["
                          << GetDataFormatStr(selected_input_format) << ", " << GetDataTypeStr(selected_input_type)
                          << "]";
#endif
                status = AddReorderOp(options, edge_id, node_id, REORDER_INPUT, input_format, selected_input_format,
                                      input_type, selected_input_type);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Add reorder op failed.";
                    return status;
                }
            }
        }

        // extra input(used by if/loop op) force to be ndarray
        for (uint32_t i = 0; i < node->GetExtraInputCount(); i++) {
            auto edge_id = node->GetExtraInput(i);
            auto extra_input_format = tensor_impls_[edge_id]->GetShape()->GetDataFormat();
            auto extra_input_type = tensor_impls_[edge_id]->GetShape()->GetDataType();
            if (extra_input_format != ppl::common::DATAFORMAT_NDARRAY) {
                status = AddReorderOp(options, edge_id, node_id, REORDER_EXTRA_INPUT, extra_input_format,
                                      ppl::common::DATAFORMAT_NDARRAY, extra_input_type, extra_input_type);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Add reorder op failed.";
                    return status;
                }
            }
        }

        for (uint32_t i = 0; i < node->GetOutputCount(); i++) {
            auto edge_id = node->GetOutput(i);
            auto selected_output_format = selected_output_formats[i];
            tensor_impls_[edge_id]->GetShape()->SetDataFormat(selected_output_format);
            kernel->SetOutputDataFormat(i, selected_output_format);
        }

        for (uint32_t i = 0; i < node->GetOutputCount(); i++) {
            auto edge_id = node->GetOutput(i);
            auto selected_output_type = selected_output_types[i];
            tensor_impls_[edge_id]->GetShape()->SetDataType(selected_output_type);
            kernel->SetOutputDataType(i, selected_output_type);
        }
    }

    return RC_SUCCESS;
}

RetCode OptGraph::TryToInferType(ArmDevice* device) {
    vector<nodeid_t> sorted_nodes;
    graph_->topo->TopologicalSort([&sorted_nodes](nodeid_t nid) -> void {
        sorted_nodes.push_back(nid);
    });

    /** get user specified forward precision */
    if (options_->forward_precision == ppl::common::DATATYPE_FLOAT16 ||
        options_->forward_precision == ppl::common::DATATYPE_FLOAT32) {
        for (uint32_t i = 0; i < graph_->topo->GetInputCount(); i++) {
            auto edge_id = graph_->topo->GetInput(i);
            const TensorShape& input_shape = *tensor_impls_[edge_id]->GetShape();
            if (input_shape.GetDataType() == ppl::common::DATATYPE_FLOAT16 ||
                input_shape.GetDataType() == ppl::common::DATATYPE_FLOAT32) {
                tensor_impls_[edge_id]->GetShape()->SetDataType(options_->forward_precision);
            }
        }
    } else {
        LOG(ERROR) << "Unsupported forward precision";
        return RC_UNSUPPORTED;
    }

    /** try to infer types for each node's input and output*/
    for (auto node_id : sorted_nodes) {
        auto node = graph_->topo->GetNode(node_id);
        bool all_inputs_has_type = true;
        for (uint32_t i = 0; i < node->GetInputCount(); i++) {
            auto input_edge = graph_->topo->GetEdge(node->GetInput(i));
            if (!input_edge) { // some op may have emtpy input
                continue;
            }
            if (tensor_impls_.find(input_edge->GetId()) == tensor_impls_.end() ||
                tensor_impls_[input_edge->GetId()]->GetShape()->GetDataType() == DATATYPE_UNKNOWN) {
                all_inputs_has_type = false;
                break;
            }
        }
        if (!all_inputs_has_type) {
            continue;
        }

        InputOutputInfo IOinfo;
        IOinfo.SetNode(node);
        IOinfo.SetAcquireFunc(acquire_tensor_func_);

        auto kernel = (ArmOptKernel*)(info_->kernels[node_id].get());
        kernel->InferTypes(&IOinfo);
        // save output data type to common param
        for (uint32_t i = 0; i < node->GetOutputCount(); i++) {
            kernel->SetOutputDataType(i, IOinfo.GetOutput<TensorImpl>(i)->GetShape()->GetDataType());
        }
    }

    return RC_SUCCESS;
}

RetCode OptGraph::TryToInferDims(ArmDevice* device) {
    vector<nodeid_t> sorted_nodes;
    graph_->topo->TopologicalSort([&sorted_nodes](nodeid_t nid) -> void {
        sorted_nodes.push_back(nid);
    });

    for (auto node_id : sorted_nodes) {
        auto node = graph_->topo->GetNode(node_id);
        bool all_inputs_has_dims = true;
        for (uint32_t i = 0; i < node->GetInputCount(); i++) {
            auto input_edge = graph_->topo->GetEdge(node->GetInput(i));
            if (!input_edge) { // some op may have emtpy input
                continue;
            }
            if (tensor_impls_.find(input_edge->GetId()) == tensor_impls_.end() ||
                tensor_impls_[input_edge->GetId()]->GetShape()->GetDimCount() == 0) {
                all_inputs_has_dims = false;
                break;
            }
        }
        if (!all_inputs_has_dims) {
            continue;
        }

        InputOutputInfo IOinfo;
        IOinfo.SetNode(node);
        IOinfo.SetAcquireFunc(acquire_tensor_func_);

        auto kernel = (ArmOptKernel*)(info_->kernels[node_id].get());
        auto status = kernel->InferDims(&IOinfo);
        if (status != RC_SUCCESS) {
            continue;
        }
    }

    return RC_SUCCESS;
}

ppl::common::RetCode OptGraph::CreateArmOptKernel(const OptKernelOptions& options, const ir::Node* node,
                                                  ArmOptKernel** kernel) {
    auto& type = node->GetType();

    auto creator = OptKernelCreatorManager::GetInstance()->Find(type.domain, type.name, type.version);
    if (!creator) {
        LOG(ERROR) << "Cannot find creator for ArmOptKernel[" << node->GetName() << "] type[" << type.domain << ":"
                   << type.name << "]";
        return RC_NOT_FOUND;
    }

    auto opt_kernel = unique_ptr<ArmOptKernel>((*creator)(node));
    if (!opt_kernel) {
        LOG(ERROR) << "Create ArmOptKernel failed: oom";
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

RetCode OptGraph::DoOptimize(const utils::SharedResource& resource, ArmDevice* device) {
    OptKernelOptions options;
    options.resource = &resource;
    options.graph_data = graph_->data.get();
    options.graph_topo = graph_->topo.get();
    options.device = device;
    options.info = info_;
    options.tensors = &tensor_impls_;
    options.engine_options = options_;

    for (auto it = info_->kernels.begin(); it != info_->kernels.end(); ++it) {
        auto kernel = (ArmOptKernel*)(it->second.get());
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
                   tensor->GetShape()->GetBytesExcludingPadding());
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

    const auto opt_rule_manager = OptRuleManager::Instance();
    const auto max_opt_level = opt_rule_manager->GetMaxOptLevel(options_->graph_optimization_level);

    // before layout optimize
    status = opt_rule_manager->ApplyRules(options, max_opt_level, "BeforeLayoutOptimize", "");
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Run BeforeLayoutOptimize failed: " << GetRetCodeStr(status);
        return status;
    }

    status = StitchGraph(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Graph Stitching failed: " << GetRetCodeStr(status);
        return status;
    }

    status = opt_rule_manager->ApplyRules(options, max_opt_level, "AfterLayoutOptimize", "");

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

}}} // namespace ppl::nn::arm
