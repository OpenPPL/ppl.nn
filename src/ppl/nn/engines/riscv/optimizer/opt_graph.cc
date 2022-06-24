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

#include <string.h>

#include "ppl/nn/engines/riscv/optimizer/opt_graph.h"
#include "ppl/nn/engines/riscv/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/riscv/optimizer/opt_rule_manager.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/utils.h"

// #define SHOW_GRAPH_VIS
#ifdef SHOW_GRAPH_VIS
#include "ppl/nn/auxtools/to_graphviz.h"
#include <fstream>
#endif

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

RetCode OptGraph::InitKernels(const ir::Graph* graph) {
    auto topo = graph->topo.get();
    for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        auto& type = node->GetType();
        auto creator = OptKernelCreatorManager::GetInstance()->Find(type.domain, type.name, type.version);
        if (!creator) {
            LOG(ERROR) << "cannot find creator for RiscvOptKernel[" << node->GetName() << "] type[" << type.domain
                       << ":" << type.name << "]";
            return RC_NOT_FOUND;
        }

        auto opt_kernel = unique_ptr<RiscvOptKernel>((*creator)(node));
        if (!opt_kernel) {
            LOG(ERROR) << "create RiscvOptKernel failed: oom";
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

RetCode OptGraph::Init(ir::Graph* graph, RuntimePartitionInfo* info, EngineOptions* options) {
    graph_ = graph;
    info_ = info;
    options_ = options;

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

    acquire_tensor_func_ = [this](edgeid_t eid, uint32_t) -> EdgeObject* {
        auto it = tensor_impls_.find(eid);
        if (it == tensor_impls_.end()) {
            return nullptr;
        }
        return it->second.get();
    };

    return RC_SUCCESS;
}

RetCode OptGraph::TryToInferType(RiscvDevice* device) {
    vector<nodeid_t> sorted_nodes;
    graph_->topo->TopologicalSort([&sorted_nodes](nodeid_t nid) -> void {
        sorted_nodes.push_back(nid);
    });

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

        auto kernel = (RiscvOptKernel*)(info_->kernels[node_id].get());
        kernel->InferType(&IOinfo);
    }

    return RC_SUCCESS;
}

RetCode OptGraph::TryToInferDims(RiscvDevice* device) {
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

        auto kernel = (RiscvOptKernel*)(info_->kernels[node_id].get());
        auto status = kernel->InferDims(&IOinfo);
        if (status != RC_SUCCESS) {
            continue;
        }
    }

    return RC_SUCCESS;
}

RetCode OptGraph::DoOptimize(const utils::SharedResource& resource, RiscvDevice* device) {
    OptKernelOptions options;
    options.resource = &resource;
    options.graph_data = graph_->data.get();
    options.graph_topo = graph_->topo.get();
    options.tensors = &tensor_impls_;
    options.device = device;
    options.info = info_;
    options.engine_options = options_;

    for (auto it = info_->kernels.begin(); it != info_->kernels.end(); ++it) {
        auto kernel = (RiscvOptKernel*)(it->second.get());
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
            memcpy(tensor->GetBufferPtr<void>(), graph_->data->constants[edge_id].data.GetData(),
                   tensor->GetShape()->CalcBytesExcludingPadding());
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

    auto opt_rule_manager = OptRuleManager::Instance();

    opt_rule_manager->ApplyByTag("BeforeLayoutOptimize", options);

    if (true != opt_rule_manager->Apply("", "LayoutOptimize", options)) {
        LOG(ERROR) << "LayoutOptimize failed";
        return ppl::common::RC_OTHER_ERROR;
    }

    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Graph Stitching failed: " << GetRetCodeStr(status);
        return status;
    }

    opt_rule_manager->ApplyByTag("AfterLayoutOptimize", options);

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

}}} // namespace ppl::nn::riscv
