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
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/bridge_op.h"
#include "ppl/nn/engines/utils.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

OptGraph::OptGraph(ir::Graph* graph, RuntimePartitionInfo* info, CudaArgs* args, CompileInfo* compile_set)
        : graph_(graph), info_(info), args_(args), compile_set_(compile_set) {
    acquire_tensor_func_ = [this](edgeid_t eid, uint32_t) -> EdgeObject* {
        auto it = tensor_impls_.find(eid);
        if (it == tensor_impls_.end()) {
            return nullptr;
        }
        return it->second.get();
    };
}

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
        auto creator = OptKernelCreatorManager::GetInstance()->Find(type.domain, type.name, type.version);
        if (!creator) {
            LOG(ERROR) << "cannot find creator for CudaOptKernel[" << node->GetName() << "] type[" << type.domain << ":"
                       << type.name << "]";
            return RC_NOT_FOUND;
        }

        auto opt_kernel = unique_ptr<CudaOptKernel>((*creator)(node));
        if (!opt_kernel) {
            LOG(ERROR) << "create CudaOptKernel failed: oom";
            return RC_OUT_OF_MEMORY;
        }

        info_->kernels.emplace(node->GetId(), std::move(opt_kernel));
    }

    return RC_SUCCESS;
}

RetCode OptGraph::UpdateDims(const utils::SharedResource& resource) {
    auto topo = graph_->topo.get();
    auto data = graph_->data.get();

    vector<nodeid_t> sorted_node_ids;
    topo->TopologicalSort([&sorted_node_ids](nodeid_t nid) -> void {
        sorted_node_ids.push_back(nid);
    });

    OptKernelOptions options(graph_, &resource);
    UpdateTopologicalSort();

    InputOutputInfo IOinfo;
    IOinfo.SetAcquireFunc(acquire_tensor_func_);

    for (uint32_t i = 0; i < sorted_node_ids_.size(); ++i) {
        auto node = topo->GetNode(sorted_node_ids_[i]);
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

            auto edge = topo->GetEdge(edge_id);
            auto impl_pair = tensor_impls_.insert(
                make_pair(edge_id, unique_ptr<TensorImpl>(new TensorImpl(edge, TENSORTYPE_NORMAL))));
            if (impl_pair.second) {
                // default shape
                TensorShape temp_tensor_shape;

                // replace to model-given shape
                auto ir_shape = data->shapes.find(edge_id);
                if (ir_shape != data->shapes.end()) {
                    utils::IrShape2TensorShape(ir_shape->second, &temp_tensor_shape);
                    auto dim_count = temp_tensor_shape.GetRealDimCount();
                    if (dim_count > 0) {
                        // replace dynamic dims to default values
                        if (temp_tensor_shape.GetDim(0) == INVALID_DIM_VALUE) {
                            temp_tensor_shape.SetDim(0, 1);
                        }
                        for (uint32_t k = 1; k < dim_count; ++k) {
                            if (temp_tensor_shape.GetDim(k) == INVALID_DIM_VALUE) {
                                temp_tensor_shape.SetDim(k, 224);
                            }
                        }
                    }
                } else {
                    temp_tensor_shape.Reshape({1, 3, 224, 224});
                    temp_tensor_shape.SetDataFormat(DATAFORMAT_NDARRAY);
                    temp_tensor_shape.SetDataType(DATATYPE_UNKNOWN);
                }

                if (topo->GetInput(edge->GetName()) != INVALID_EDGEID) { // input j is a graph input edge
                    if (j < args_->input_dims.size() && !args_->input_dims[j].empty()) { // args include input shape
                        const vector<int64_t>* dims = &args_->input_dims[j];
                        temp_tensor_shape.SetDimCount(dims->size());
                        for (uint32_t k = 0; k < dims->size(); ++k) {
                            temp_tensor_shape.SetDim(k, dims->at(k));
                        }
                    }
                }

                *impl_pair.first->second->GetShape() = temp_tensor_shape;
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
            auto edge = topo->GetEdge(edge_id);
            tensor_impls_.insert(make_pair(edge_id, unique_ptr<TensorImpl>(new TensorImpl(edge, TENSORTYPE_NORMAL))));
        }

        status = kernel->InferDims(&IOinfo);
        if (status != RC_SUCCESS) {
            LOG(DEBUG) << "Can not reshape safely for node[" << node->GetName() << "]";
            illegal_dims_.emplace(sorted_node_ids_[i]);
            std::set<uint32_t> illegal_inputs;
            for (uint32_t j = 0; j < node->GetInputCount(); ++j) {
                auto preedge_id = node->GetInput(j);
                auto prenode_id = topo->GetEdge(preedge_id)->GetProducer();
                if (illegal_dims_.find(prenode_id) == illegal_dims_.end()) {
                    illegal_inputs.emplace(j);
                }
            }
            kernel->InferUnsafeDims(&IOinfo, &illegal_inputs);
        }
    }

    LOG(DEBUG) << "Create " << tensor_impls_.size() << " TensorImpl";
    return RC_SUCCESS;
}

RetCode OptGraph::FuseOperator(const utils::SharedResource& resource) {
    ir::GraphTopo* topo = graph_->topo.get();
    UpdateTopologicalSort();
    auto fs_filter_manager = FsFilterManager::Instance();

    OptKernelOptions options(graph_, info_, &resource, &tensor_impls_);
    int32_t index = LastLegalNodeIndex();

    for (int32_t i = sorted_node_ids_.size() - 1; i >= 0; --i) {
        auto node = topo->GetNode(sorted_node_ids_[i]);
        if (node) {
            auto fuse = fs_filter_manager->FindFusion(node->GetType().name);
            if (fuse) {
                fuse->FuseNode(node, i <= index, options);
            }
        }
    }
    return RC_SUCCESS;
}

RetCode OptGraph::AddBridgeKernels(const utils::SharedResource& resource) {
    auto topo = graph_->topo.get();
    auto& tensor_params = args_->quant_info.tensor_params;
    uint32_t count = 0;
    OptKernelOptions options(graph_, &resource);

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
            auto edge = topo->GetEdge(edge_id);
            if (edge->GetName().find("Bridge_Edge") != string::npos) {
                continue;
            }

            auto creator = OptKernelCreatorManager::GetInstance()->Find("pmx", "Bridge", 1);
            auto ret_pair = topo->AddNode("Bridge_Node_" + node->GetName() + "_" + edge->GetName());
            if (!ret_pair.second) {
                LOG(ERROR) << "create a new node for [" << edge->GetName() << "] failed.";
                return RC_OUT_OF_MEMORY;
            }
            auto new_node = ret_pair.first;

            new_node->SetType(ir::Node::Type("pmx", "Bridge", 1));
            auto bridge_kernel = unique_ptr<CudaOptKernel>((*creator)(new_node));
            ((BridgeOp*)bridge_kernel.get())->AddInternalBridgeNode(node, new_node, edge, graph_);

            auto preedge_id = new_node->GetInput(0);
            auto postedge_id = new_node->GetOutput(0);
            auto new_edge = topo->GetEdge(postedge_id);
            auto impl_pair = tensor_impls_.insert(
                make_pair(postedge_id, unique_ptr<TensorImpl>(new TensorImpl(new_edge, TENSORTYPE_NORMAL))));
            auto pre_shape = tensor_impls_.find(preedge_id)->second.get();
            impl_pair.first->second->GetShape()->Reshape(pre_shape->GetShape()->GetDims(),
                                                        pre_shape->GetShape()->GetRealDimCount());

            bridge_kernel.get()->Init(options);
            info_->kernels.emplace(new_node->GetId(), std::move(bridge_kernel));
            auto tensor_pair = tensor_params.find(edge->GetName());
            if (tensor_pair != tensor_params.end())
                tensor_params.emplace(new_edge->GetName(), tensor_pair->second);
            count++;
        }

        for (uint32_t j = 0; j < node->GetOutputCount(); ++j) {
            auto edge = topo->GetEdge(node->GetOutput(j));
            if (topo->GetOutput(edge->GetName()) != INVALID_EDGEID || // it is marked as an output node
                edge->CalcConsumerCount() == 0) { // it is an finel node for the graph
                auto creator = OptKernelCreatorManager::GetInstance()->Find("pmx", "Bridge", 1);

                auto ret_pair = topo->AddNode("Bridge_Final_" + node->GetName() + "_" + edge->GetName());
                if (!ret_pair.second) {
                    LOG(ERROR) << "create a new node for [" << edge->GetName() << "] failed.";
                    return RC_OUT_OF_MEMORY;
                }
                auto new_node = ret_pair.first;

                new_node->SetType(ir::Node::Type("pmx", "Bridge", 1));
                auto bridge_kernel = unique_ptr<CudaOptKernel>((*creator)(new_node));
                ((BridgeOp*)bridge_kernel.get())->AddFinalBridgeNode(node, new_node, edge, graph_);

                auto preedge_id = new_node->GetInput(0);
                auto postedge_id = new_node->GetOutput(0);
                auto new_edge = topo->GetEdge(preedge_id);
                auto impl_pair = tensor_impls_.insert(
                    make_pair(preedge_id, unique_ptr<TensorImpl>(new TensorImpl(new_edge, TENSORTYPE_NORMAL))));
                auto post_shape = tensor_impls_.find(postedge_id)->second.get();

                impl_pair.first->second->GetShape()->Reshape(post_shape->GetShape()->GetDims(),
                                                            post_shape->GetShape()->GetRealDimCount());

                if (j < args_->output_formats.size()) {
                    post_shape->GetShape()->SetDataFormat(args_->output_formats[j]);
                } else {
                    post_shape->GetShape()->SetDataFormat(DATAFORMAT_NDARRAY);
                }

                bridge_kernel.get()->Init(options);
                info_->kernels.emplace(new_node->GetId(), std::move(bridge_kernel));
                auto tensor_pair = tensor_params.find(edge->GetName());
                if (tensor_pair != tensor_params.end())
                    tensor_params.emplace(new_edge->GetName(), tensor_pair->second);
                count++;
            }
        }
    }

    LOG(INFO) << "added " << count << " new bridge kernels";
    return RC_SUCCESS;
}

RetCode OptGraph::InitQuantization() {
    auto topo = graph_->topo.get();
    std::vector<CudaTensorQuant> graph_quants(topo->GetCurrentEdgeIdBound());

    // Load node quant to args_->node_type
    auto& node_params = args_->quant_info.node_params;
    for (auto iter = topo->CreateNodeIter(); iter->IsValid(); iter->Forward()) {
        auto node = iter->Get();
        auto pair = node_params.find(node->GetName());
        if (pair != node_params.end()) {
            auto str = pair->second.fields.find("data_type")->second;
            if (str.content == "INT8") {
                args_->node_types.emplace(node->GetName(), DATATYPE_INT8);
            } else if (str.content == "FLOAT32") {
                args_->node_types.emplace(node->GetName(), DATATYPE_FLOAT32);
            } else {
                LOG(ERROR) << "Not support set to such datatype: " << str.content;
            }
        }
    }
    // Load tensor quant to args_->quant_info
    auto& tensor_params = args_->quant_info.tensor_params;
    for (auto iter = topo->CreateEdgeIter(); iter->IsValid(); iter->Forward()) {
        auto edge = iter->Get();
        auto pair = tensor_params.find(edge->GetName());
        // Can not find quant info. It means quant info is not exist.
        if (pair == tensor_params.end()) {
            continue;
        }
        auto& temp_tensor_quant = graph_quants[edge->GetId()];
        auto str = pair->second.fields.find("per_channel")->second;
        temp_tensor_quant.per_channel = *(bool*)(str.content.data());
        auto bit_width = pair->second.fields.find("bit_width")->second;
        temp_tensor_quant.bit_width = *(int*)(bit_width.content.data());
        if (temp_tensor_quant.per_channel) {
            auto max_str = pair->second.fields.find("tensor_max")->second;
            auto min_str = pair->second.fields.find("tensor_min")->second;
            auto scale_str = pair->second.fields.find("scale")->second;
            uint32_t size = max_str.content.length() / 8;
            temp_tensor_quant.scale.resize(size);
            temp_tensor_quant.zero_point.resize(size);
            for (uint32_t i = 0; i < size; ++i) {
                auto tensor_max = *((double*)(max_str.content.data()) + i);
                auto tensor_min = *((double*)(min_str.content.data()) + i);
                temp_tensor_quant.scale[i] = (double)(tensor_max - tensor_min) / ((1 << temp_tensor_quant.bit_width) - 1);
                temp_tensor_quant.zero_point[i] = tensor_max + tensor_min;
            }
        } else {
            str = pair->second.fields.find("tensor_max")->second;
            auto tensor_max = *(double*)(str.content.data());
            str = pair->second.fields.find("tensor_min")->second;
            auto tensor_min = *(double*)(str.content.data());
            auto scale_str = pair->second.fields.find("scale")->second;
            temp_tensor_quant.scale[0] = (double)(tensor_max - tensor_min) / ((1 << temp_tensor_quant.bit_width) - 1);
            temp_tensor_quant.zero_point[0] = tensor_max + tensor_min;
        }
    }

    args_->tensor_quants.emplace(topo->GetName(), std::move(graph_quants));
    return RC_SUCCESS;
}

RetCode OptGraph::UpdateType() {
    auto topo = graph_->topo.get();
    auto& graph_quants = args_->tensor_quants.find(topo->GetName())->second;
    UpdateTopologicalSort();

    InputOutputInfo IOinfo;
    IOinfo.SetAcquireFunc(acquire_tensor_func_);

    for (uint32_t i = 0; i < sorted_node_ids_.size(); ++i) {
        auto node = topo->GetNode(sorted_node_ids_[i]);

        IOinfo.SetNode(node);
        CudaOptKernel* kernel = (CudaOptKernel*)(info_->kernels.find(node->GetId())->second.get());

        datatype_t kernel_type = args_->default_kernel_type;
        auto conf_pair = args_->node_types.find(node->GetName());
        if (conf_pair != args_->node_types.end()) {
            kernel_type = conf_pair->second;
        }
        if (kernel_type == DATATYPE_INT8) {
            for (uint32_t i = 0; i < node->GetInputCount(); ++i) {
                auto edge_id = node->GetInput(i);
                if (edge_id == INVALID_EDGEID) {
                    continue;
                }
                auto& input_quant = graph_quants.at(edge_id);
                input_quant.type = kernel_type;
            }
            for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
                auto edge_id = node->GetOutput(i);
                if (edge_id == INVALID_EDGEID) {
                    continue;
                }
                auto& output_quant = graph_quants.at(edge_id);
                output_quant.type = kernel_type;
            }
        }
        auto status = kernel->InferType(&IOinfo, &graph_quants, kernel_type);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Set type for node[" << node->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }

        // it is an output node
        for (uint32_t j = 0; j < node->GetOutputCount(); ++j) {
            auto edge = topo->GetEdge(node->GetOutput(j));
            if (edge->CalcConsumerCount() == 0) {
                auto out_shape = IOinfo.GetOutput<TensorImpl>(j)->GetShape();
                if (out_shape->GetDataType() == DATATYPE_FLOAT16 || out_shape->GetDataType() == DATATYPE_INT8)
                    out_shape->SetDataType(DATATYPE_FLOAT32);

                if (j < args_->output_types.size()) {
                    out_shape->SetDataType(args_->output_types[j]);
                }
            }
        }
    }

    return RC_SUCCESS;
}

RetCode OptGraph::SelectAlgos(const utils::SharedResource& resource, CudaDevice* device) {
    auto topo = graph_->topo.get();
    auto& graph_quants = args_->tensor_quants.find(topo->GetName())->second;
    auto& graph_algos = args_->alog_selects;

    OptKernelOptions options(graph_, info_, &resource, args_, compile_set_, device, &tensor_impls_, &graph_quants, &graph_algos);
    UpdateTopologicalSort();

    AlgoGraph algo_graph(topo);
    // calculate the least time consuming
    for (uint32_t i = 0; i < sorted_node_ids_.size(); ++i) {
        auto node = topo->GetNode(sorted_node_ids_[i]);
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
        auto node = topo->GetNode(node_id);

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
    auto& graph_quants = args_->tensor_quants.find(topo->GetName())->second;

    for (auto iter = topo->CreateNodeIter(); iter->IsValid(); iter->Forward()) {
        auto node = iter->Get();
        if (node->GetType().name != "Bridge") {
            continue;
        }

        auto preedge_id = node->GetInput(0);
        auto postedge_id = node->GetOutput(0);
        const TensorShape& preshape = *tensor_impls_.find(preedge_id)->second->GetShape();
        const TensorShape& postshape = *tensor_impls_.find(postedge_id)->second->GetShape();

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

            auto converter = (CudaDataConverter*)device->GetDataConverter();
            status = converter->ConvertFromHost(&constant_info.GetBufferDesc(), postshape, graph_quants[postedge_id],
                                                constant_ref->second.data.data(), preshape, graph_quants[preedge_id]);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "copy constant failed: " << GetRetCodeStr(status);
                return status;
            }

            info_->constants[preedge_id] = std::move(constant_info);
            *tensor_impls_.find(preedge_id)->second->GetShape() = postshape;
            graph_quants[preedge_id] = graph_quants[postedge_id];
        }
    }

    // load the rest of constants that are not quantized
    auto status = utils::LoadConstants(*graph_, device, &info_->constants);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load constants failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode OptGraph::DeleteBridgeKernels() {
    auto topo = graph_->topo.get();
    auto& graph_quants = args_->tensor_quants.find(topo->GetName())->second;
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
        auto status =
            ((BridgeOp*)(bridge_kernel->second.get()))->DeleteBridgeNode(node, graph_, &tensor_impls_, &graph_quants);
        if (status == RC_SUCCESS) {
            info_->kernels.erase(node_id);
            count++;
        }
    }

    LOG(INFO) << "deleted " << count << " bridge kernels";
    return RC_SUCCESS;
}

RetCode OptGraph::DoOptimize(const utils::SharedResource& resource, CudaDevice* device) {
    auto status = InitKernels();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init kernels failed: " << GetRetCodeStr(status);
        return status;
    }

    status = UpdateDims(resource);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Update dims failed: " << GetRetCodeStr(status);
        return status;
    }

    status = FuseOperator(resource);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Fuse operators failed: " << GetRetCodeStr(status);
        return status;
    }

    status = AddBridgeKernels(resource);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Add Bridge nodes failed: " << GetRetCodeStr(status);
        return status;
    }

    status = InitQuantization();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init quantization failed: " << GetRetCodeStr(status);
        return status;
    }

    status = UpdateType();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Update type failed: " << GetRetCodeStr(status);
        return status;
    }

    status = SelectAlgos(resource, device);
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

    status = FuseOperator(resource);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Fuse operators failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
