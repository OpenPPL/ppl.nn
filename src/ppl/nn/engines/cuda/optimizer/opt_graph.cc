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

#include "ppl/nn/engines/cuda/optimizer/opt_graph.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/engine.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/cuda/optimizer/algos/algo_graph.h"
#include "ppl/nn/engines/cuda/optimizer/ops/ppl/bridge_op.h"
#include "ppl/nn/engines/utils.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

OptGraph::~OptGraph() {
    // destroy tensors before default_cpu_device_
    tensor_impls_.clear();
}

int32_t OptGraph::LastLegalNodeIndex() {
    for (uint32_t i = 0; i < sorted_node_ids_.size(); ++i) {
        if (illegal_dims_.find(sorted_node_ids_[i]) != illegal_dims_.end()) {
            return (int32_t)i;
        }
    }
    return sorted_node_ids_.size() - 1;
}

void OptGraph::UpdateTopologicalSort() {
    ir::GraphTopo* topo = graph_->topo.get();
    sorted_node_ids_.clear();
    topo->TopologicalSort([this](nodeid_t nid) -> void {
        sorted_node_ids_.push_back(nid);
    });
}

RetCode OptGraph::InitKernels() {
    auto topo = graph_->topo.get();
    for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        auto& type = node->GetType();
        auto creator = OptKernelCreatorManager::Instance()->Find(type.domain, type.name);
        if (!creator) {
            LOG(ERROR) << "cannot find creator for CudaOptKernel[" << node->GetName() << "] type[" << type.domain << ":"
                       << type.name << "]";
            return RC_NOT_FOUND;
        }

        auto opt_kernel = unique_ptr<CudaOptKernel>(creator(node));
        if (!opt_kernel) {
            LOG(ERROR) << "create CudaOptKernel failed: oom";
            return RC_OUT_OF_MEMORY;
        }

        info_->kernels.emplace(node->GetId(), std::move(opt_kernel));
    }

    return RC_SUCCESS;
}

RetCode OptGraph::UpdateDims() {
    ir::GraphTopo* topo = graph_->topo.get();
    ir::GraphData* data = graph_->data.get();

    vector<nodeid_t> sorted_node_ids;
    topo->TopologicalSort([&sorted_node_ids](nodeid_t nid) -> void {
        sorted_node_ids.push_back(nid);
    });

    OptKernelOptions options(graph_, resource_);
    UpdateTopologicalSort();

    InputOutputInfo IOinfo;
    IOinfo.SetAcquireObjectFunc([this](edgeid_t eid, uint32_t, Device*) -> EdgeObject* {
        auto iter = tensor_impls_.find(eid);
        if (iter == tensor_impls_.end()) {
            return nullptr;
        }
        return iter->second.get();
    });

    for (uint32_t i = 0; i < sorted_node_ids_.size(); ++i) {
        auto node = topo->GetNodeById(sorted_node_ids_[i]);
        IOinfo.SetNode(node);

        CudaOptKernel* kernel = (CudaOptKernel*)(info_->kernels.find(node->GetId())->second.get());
        auto status = kernel->Init(options);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Init kernel failed.";
            return RC_UNSUPPORTED;
        }

        for (uint32_t j = 0; j < node->GetInputCount(); ++j) {
            auto edge_id = node->GetInput(j);
            if (edge_id == INVALID_EDGEID) {
                continue;
            }

            auto edge = topo->GetEdgeById(edge_id);
            if (topo->GetInput(edge->GetName()) != INVALID_EDGEID &&
                topo->GetConstant(edge->GetName()) == INVALID_EDGEID) { // input j is a graph input edge
                auto ir_shape = data->shapes.find(edge_id);
                if (ir_shape == data->shapes.end()) {
                    LOG(ERROR) << "cannot find input shape in data map.";
                    return RC_INVALID_VALUE;
                }
                auto dims_pairs = args_->input_dims.find(edge->GetName());
                if (dims_pairs == args_->input_dims.end()) { // use default dims
                    dims_pairs = args_->input_dims.find("");
                    if (dims_pairs == args_->input_dims.end()) {
                        LOG(ERROR) << "Error input dims init for input edge[" << edge->GetName() << "]";
                        return RC_INVALID_VALUE;
                    }
                }
                if (ir_shape->second.dims.size() == dims_pairs->second.size()) {
                    for (uint32_t k = 0; k < ir_shape->second.dims.size(); ++k) {
                        if (ir_shape->second.dims[k] == 1 && (dims_pairs->second)[k] != 0) {
                            ir_shape->second.dims[k] = (dims_pairs->second)[k];
                        }
                    }
                }
            }

            auto impl_pair = tensor_impls_.insert(
                make_pair(edge_id, unique_ptr<TensorImpl>(new TensorImpl(edge, TENSORTYPE_NORMAL))));
            if (impl_pair.second) {
                // default shape
                TensorShape temp_tensor_shape;
                temp_tensor_shape.Reshape({1, 3, 128, 128});
                temp_tensor_shape.SetDataFormat(DATAFORMAT_NDARRAY);
                temp_tensor_shape.SetDataType(DATATYPE_UNKNOWN);

                // replace to model-given shape
                auto ir_shape = data->shapes.find(edge_id);
                if (ir_shape != data->shapes.end()) {
                    utils::IrShape2TensorShape(ir_shape->second, &temp_tensor_shape);
                }

                impl_pair.first->second->GetShape() = temp_tensor_shape;
                auto constant_ref = graph_->data->constants.find(edge_id);
                if (constant_ref != graph_->data->constants.end()) { // constant tensor
                    auto tensor = impl_pair.first->second.get();
                    tensor->SetDevice(&default_cpu_device_);

                    auto status = tensor->ReallocBuffer();
                    if (status != RC_SUCCESS) {
                        LOG(ERROR) << "realloc buffer failed: " << GetRetCodeStr(status);
                        return status;
                    }

                    status = tensor->CopyFromHost(constant_ref->second.data.data());
                    if (status != RC_SUCCESS) {
                        LOG(ERROR) << "copy constant [" << tensor->GetName() << "] failed: " << GetRetCodeStr(status);
                        return status;
                    }
                }
            }
        }

        for (uint32_t j = 0; j < node->GetOutputCount(); ++j) {
            auto edge_id = node->GetOutput(j);
            if (edge_id == INVALID_EDGEID) {
                continue;
            }
            auto edge = topo->GetEdgeById(edge_id);
            tensor_impls_.insert(make_pair(edge_id, unique_ptr<TensorImpl>(new TensorImpl(edge, TENSORTYPE_NORMAL))));
        }

        status = kernel->InferDims(&IOinfo);
        if (status != RC_SUCCESS) {
            LOG(DEBUG) << "Can not reshape safely for node[" << node->GetName() << "]";
            illegal_dims_.emplace(sorted_node_ids_[i]);
            std::set<uint32_t> illegal_inputs;
            for (uint32_t j = 0; j < node->GetInputCount(); ++j) {
                auto preedge_id = node->GetInput(j);
                auto prenode_id = topo->GetEdgeById(preedge_id)->GetProducer();
                if (illegal_dims_.find(prenode_id) == illegal_dims_.end()) {
                    illegal_inputs.emplace(j);
                }
            }
            kernel->InferUnsafeDims(&IOinfo, &illegal_inputs);
        }
    }

    LOG(INFO) << "Create " << tensor_impls_.size() << " TensorImpl";
    return RC_SUCCESS;
}

RetCode OptGraph::FuseOperator() {
    ir::GraphTopo* topo = graph_->topo.get();
    UpdateTopologicalSort();
    auto fs_filter_manager = FsFilterManager::Instance();

    OptKernelOptions options(graph_, info_, resource_, &tensor_impls_);
    int32_t index = LastLegalNodeIndex();

    for (int32_t i = sorted_node_ids_.size() - 1; i >= 0; --i) {
        auto node = topo->GetNodeById(sorted_node_ids_[i]);
        if (node) {
            auto fuse = fs_filter_manager->FindFusion(node->GetType().name);
            if (fuse) {
                fuse->FuseNode(node, i <= index, options);
            }
        }
    }
    return RC_SUCCESS;
}

RetCode OptGraph::AddBridgeKernels() {
    auto topo = graph_->topo.get();
    uint32_t count = 0;
    OptKernelOptions options(graph_, resource_);

    for (auto iter = topo->CreateNodeIter(); iter->IsValid(); iter->Forward()) {
        auto node = iter->Get();

        if (node->GetType().name == "Bridge") {
            continue;
        }

        for (uint32_t j = 0; j < node->GetInputCount(); ++j) {
            auto edge_id = node->GetInput(j);
            if (edge_id == INVALID_EDGEID) {
                continue;
            }

            auto edge = topo->GetEdgeById(edge_id);
            if (edge->GetName().find("Bridge_Edge") != string::npos) {
                continue;
            }
            auto creator = OptKernelCreatorManager::Instance()->Find("ppl", "Bridge");
            auto ret_pair = topo->AddNode("Bridge_Node_" + node->GetName() + "_" + edge->GetName());
            if (!ret_pair.second) {
                LOG(ERROR) << "create a new node for [" << edge->GetName() << "] failed.";
                return RC_OUT_OF_MEMORY;
            }
            auto new_node = ret_pair.first;

            new_node->SetType(ir::Node::Type("ppl", "Bridge"));
            auto bridge_kernel = unique_ptr<CudaOptKernel>(creator(new_node));
            ((BridgeOp*)bridge_kernel.get())->AddInternalBridgeNode(node, new_node, edge, graph_);

            auto postedge_id = new_node->GetOutput(0);
            auto impl_pair = tensor_impls_.insert(
                make_pair(postedge_id, unique_ptr<TensorImpl>(new TensorImpl(edge, TENSORTYPE_NORMAL))));
            auto pre_shape = tensor_impls_.find(new_node->GetInput(0))->second.get();
            impl_pair.first->second->GetShape().Reshape(pre_shape->GetShape().GetDims(),
                                                        pre_shape->GetShape().GetRealDimCount());

            bridge_kernel.get()->Init(options);
            info_->kernels.emplace(new_node->GetId(), std::move(bridge_kernel));
            count++;
        }

        for (uint32_t j = 0; j < node->GetOutputCount(); ++j) {
            auto edge = topo->GetEdgeById(node->GetOutput(j));
            if (topo->GetOutput(edge->GetName()) != INVALID_EDGEID || // it is marked as an output node
                edge->CalcConsumerCount() == 0) { // it is an finel node for the graph
                auto creator = OptKernelCreatorManager::Instance()->Find("ppl", "Bridge");

                auto ret_pair = topo->AddNode("Bridge_Final_" + node->GetName() + "_" + edge->GetName());
                if (!ret_pair.second) {
                    LOG(ERROR) << "create a new node for [" << edge->GetName() << "] failed.";
                    return RC_OUT_OF_MEMORY;
                }
                auto new_node = ret_pair.first;

                new_node->SetType(ir::Node::Type("ppl", "Bridge"));
                auto bridge_kernel = unique_ptr<CudaOptKernel>(creator(new_node));
                ((BridgeOp*)bridge_kernel.get())->AddFinalBridgeNode(node, new_node, edge, graph_);

                auto preedge_id = new_node->GetInput(0);
                auto new_edge = topo->GetEdgeById(preedge_id);
                auto impl_pair = tensor_impls_.insert(
                    make_pair(preedge_id, unique_ptr<TensorImpl>(new TensorImpl(new_edge, TENSORTYPE_NORMAL))));
                auto post_shape = tensor_impls_.find(new_node->GetOutput(0))->second.get();

                impl_pair.first->second->GetShape().Reshape(post_shape->GetShape().GetDims(),
                                                            post_shape->GetShape().GetRealDimCount());

                auto pair_format = args_->output_formats.find(edge->GetName());
                if (pair_format != args_->output_formats.end()) {
                    post_shape->GetShape().SetDataFormat(pair_format->second);
                } else {
                    post_shape->GetShape().SetDataFormat(DATAFORMAT_NDARRAY);
                }
                bridge_kernel.get()->Init(options);
                info_->kernels.emplace(new_node->GetId(), std::move(bridge_kernel));
                count++;
            }
        }
    }

    LOG(INFO) << "added " << count << " new bridge kernels";
    return RC_SUCCESS;
}

RetCode OptGraph::UpdateType() {
    ir::GraphTopo* topo = graph_->topo.get();
    UpdateTopologicalSort();

    InputOutputInfo IOinfo;
    IOinfo.SetAcquireObjectFunc([this](edgeid_t eid, uint32_t, Device*) -> EdgeObject* {
        auto iter = tensor_impls_.find(eid);
        if (iter == tensor_impls_.end()) {
            return nullptr;
        }
        return iter->second.get();
    });

    for (uint32_t i = 0; i < sorted_node_ids_.size(); ++i) {
        auto node = topo->GetNodeById(sorted_node_ids_[i]);

        IOinfo.SetNode(node);
        CudaOptKernel* kernel = (CudaOptKernel*)(info_->kernels.find(node->GetId())->second.get());

        datatype_t kernel_type = args_->kernel_default_type;
        auto conf_pair = args_->node_types.find(node->GetName());
        if (conf_pair != args_->node_types.end()) {
            kernel_type = conf_pair->second;
        }
        auto status = kernel->InferType(&IOinfo, kernel_type);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Set type for node[" << node->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }

        // it is an output node
        auto edge = topo->GetEdgeById(node->GetOutput(0));
        if (edge->CalcConsumerCount() == 0) {
            auto out_shape = &IOinfo.GetOutput<TensorImpl>(0)->GetShape();
            if (out_shape->GetDataType() == DATATYPE_FLOAT16)
                out_shape->SetDataType(DATATYPE_FLOAT32);
            auto pair_type = args_->output_types.find(edge->GetName());
            if (pair_type != args_->output_types.end()) {
                out_shape->SetDataType(pair_type->second);
            }
        }
    }

    return RC_SUCCESS;
}

RetCode OptGraph::SelectAlgos(CudaDevice* device) {
    auto topo = graph_->topo.get();
    OptKernelOptions options(graph_, info_, resource_, args_, device, &tensor_impls_);
    UpdateTopologicalSort();

    if (!PPLCudaComputeCapabilityEqual(7, 5, device->GetDeviceId())) {
        LOG(ERROR) << "PPL is not support your GPU device right now.";
        return RC_UNSUPPORTED;
    }

    AlgoGraph algo_graph(topo);
    // calculate the least time consuming
    for (uint32_t i = 0; i < sorted_node_ids_.size(); ++i) {
        auto node = topo->GetNodeById(sorted_node_ids_[i]);
        CudaOptKernel* kernel = (CudaOptKernel*)(info_->kernels.find(node->GetId())->second.get());

        auto status = algo_graph.CreateNode(node, kernel);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Create the node[" << node->GetName() << "] failed." << GetRetCodeStr(status);
            return status;
        }
        status = algo_graph.UpdateNode(node, options);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Update the node[" << node->GetName() << "] failed." << GetRetCodeStr(status);
            return status;
        }
    }

    // select algorithm method and its format
    for (int32_t i = sorted_node_ids_.size() - 1; i >= 0; --i) {
        auto node_id = sorted_node_ids_[i];
        auto node = topo->GetNodeById(node_id);

        auto kernel = info_->kernels.find(node_id);
        if (kernel == info_->kernels.end()) {
            LOG(ERROR) << "Can not find kernel[" << node->GetName() << "].";
            return RC_NOT_FOUND;
        }
        auto status = algo_graph.DetermineNode((CudaOptKernel*)(kernel->second.get()), options);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Determine the node[" << node->GetName() << "] failed." << GetRetCodeStr(status);
            return status;
        }
    }

    algo_graph.Delete();
    return RC_SUCCESS;
}

RetCode OptGraph::LoadConstants(CudaDevice* device) {
    auto topo = graph_->topo.get();
    auto graph_data = graph_->data.get();

    for (auto iter = topo->CreateNodeIter(); iter->IsValid(); iter->Forward()) {
        auto node = iter->Get();
        if (node->GetType().name != "Bridge") {
            continue;
        }

        auto preedge_id = node->GetInput(0);
        auto postedge_id = node->GetOutput(0);

        auto preshape = tensor_impls_.find(preedge_id)->second->GetShape();
        auto postshape = tensor_impls_.find(postedge_id)->second->GetShape();

        auto constant_ref = graph_data->constants.find(preedge_id);
        if (constant_ref != graph_data->constants.end() &&
            info_->constants.find(preedge_id) == info_->constants.end()) { // constant tensor
            RuntimeConstantInfo constant_info;
            constant_info.SetDevice(device);
            constant_info.Reshape(postshape);

            auto status = constant_info.ReallocBuffer();
            if (status != RC_SUCCESS && postshape.GetBytesIncludingPadding() > 0) {
                LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                return status;
            }

            status = device->GetDataConverter()->ConvertFromHost(&constant_info.GetBufferDesc(), postshape,
                                                                 constant_ref->second.data.data(), preshape);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "copy constant failed: " << GetRetCodeStr(status);
                return status;
            }

            info_->constants[preedge_id] = std::move(constant_info);
            tensor_impls_.find(preedge_id)->second->GetShape() = postshape;
        }
    }
    return RC_SUCCESS;
}

RetCode OptGraph::DeleteBridgeKernels() {
    auto topo = graph_->topo.get();
    uint32_t count = 0;

    for (auto iter = topo->CreateNodeIter(); iter->IsValid(); iter->Forward()) {
        auto node = iter->Get();

        if (node->GetType().name != "Bridge") {
            continue;
        }

        auto bridge_kernel = info_->kernels.find(node->GetId());
        if (bridge_kernel == info_->kernels.end()) {
            LOG(ERROR) << "cannot find bridge kernel for node[" << node->GetName() << "]";
            return RC_NOT_FOUND;
        }

        auto node_id = node->GetId();
        auto status = ((BridgeOp*)(bridge_kernel->second.get()))->DeleteBridgeNode(node, graph_, &tensor_impls_);
        if (status == RC_SUCCESS) {
            info_->kernels.erase(node_id);
            count++;
        }
    }

    LOG(INFO) << "deleted " << count << " bridge kernels";
    return RC_SUCCESS;
}

RetCode OptGraph::DoOptimize(CudaDevice* device) {
    auto status = InitKernels();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init kernels failed: " << GetRetCodeStr(status);
        return status;
    }

    status = UpdateDims();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Update dims failed: " << GetRetCodeStr(status);
        return status;
    }

    status = FuseOperator();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Fuse operators failed: " << GetRetCodeStr(status);
        return status;
    }

    status = AddBridgeKernels();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Add Bridge nodes failed: " << GetRetCodeStr(status);
        return status;
    }

    status = UpdateType();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Update type failed: " << GetRetCodeStr(status);
        return status;
    }

    status = SelectAlgos(device);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Selec algos for each kernel failed: " << GetRetCodeStr(status);
        return status;
    }

    status = LoadConstants(device);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Load constant tensors failed: " << GetRetCodeStr(status);
        return status;
    }

    status = DeleteBridgeKernels();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Delete some of bridge nodes failed: " << GetRetCodeStr(status);
        return status;
    }

    status = FuseOperator();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Fuse operators failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
