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

#include "ppl/nn/models/onnx/generated/onnx.pb.h"
#include "ppl/nn/models/onnx/serializer.h"
#include "ppl/nn/models/onnx/utils.h"
#include "ppl/nn/models/onnx/param_parser_manager.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/common/tensor_shape.h" // INVALID_DIM_VALUE
#include <fstream>
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

static void PackModelInfo(::onnx::ModelProto* pb_model, const map<string, uint64_t>& opset) {
    pb_model->set_ir_version(::onnx::IR_VERSION);
    pb_model->set_producer_name("pplnn");
    pb_model->set_producer_version(PPLNN_COMMIT_STR);

    for (auto o = opset.begin(); o != opset.end(); ++o) {
        auto pb_opset = pb_model->add_opset_import();
        pb_opset->set_domain(o->first);
        if (o->first.empty() && o->second < 11) {
            pb_opset->set_version(11); // will convert opset <=11 to 11
        } else {
            pb_opset->set_version(o->second);
        }
    }
}

static RetCode PackGraphNode(::onnx::GraphProto* pb_graph, const ir::Graph& graph) {
    auto topo = graph.topo.get();
    auto data = graph.data.get();

    for (auto iter = topo->CreateNodeIter(); iter->IsValid(); iter->Forward()) {
        auto node = iter->Get();
        auto pb_node = pb_graph->add_node();

        for (uint32_t i = 0; i < node->GetInputCount(); ++i) {
            auto edge = topo->GetEdge(node->GetInput(i));
            if (edge) {
                pb_node->add_input(edge->GetName());
            } else {
                pb_node->add_input("");
            }
        }
        for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
            auto edge = topo->GetEdge(node->GetOutput(i));
            pb_node->add_output(edge->GetName());
        }

        auto& node_type = node->GetType();
        pb_node->set_name(node->GetName());
        pb_node->set_op_type(node_type.name);
        pb_node->set_domain(node_type.domain);

        auto parser_info = ParamParserManager::GetInstance()->Find(node_type.domain, node_type.name, node_type.version);
        if (!parser_info) {
            LOG(ERROR) << "unsupported op: domain[" << node_type.domain << "], type[" << node_type.name << "], version["
                       << node_type.version << "]";
            return RC_UNSUPPORTED;
        }

        if (!parser_info->pack_param) {
            continue;
        }

        auto param_ref = data->attrs.find(node->GetId());
        if (param_ref == data->attrs.end()) {
            LOG(ERROR) << "cannot find param of node[" << node->GetName() << "]";
            return RC_NOT_FOUND;
        }

        auto status = parser_info->pack_param(node, param_ref->second.get(), pb_node);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "serialize param of node[" << node->GetName() << "] type[" << node_type.domain << ":"
                       << node_type.name << ":" << node_type.version << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

static RetCode PackGraphInitializer(::onnx::GraphProto* pb_graph, const ir::Graph& graph) {
    auto topo = graph.topo.get();
    auto data = graph.data.get();

    for (uint32_t i = 0; i < topo->GetConstantCount(); ++i) {
        auto eid = topo->GetConstant(i);
        auto edge = topo->GetEdge(eid);

        auto shape_ref = data->shapes.find(eid);
        if (shape_ref == data->shapes.end()) {
            LOG(ERROR) << "cannot find shape of constant[" << edge->GetName() << "]";
            return RC_NOT_FOUND;
        }

        auto constant_ref = data->constants.find(eid);
        if (constant_ref == data->constants.end()) {
            LOG(ERROR) << "cannot find data of constant[" << edge->GetName() << "]";
            return RC_NOT_FOUND;
        }

        auto pb_tensor = pb_graph->add_initializer();
        pb_tensor->set_name(edge->GetName());

        auto status = utils::PackTensorProto(constant_ref->second.data.GetData(), constant_ref->second.data.GetSize(),
                                             shape_ref->second.data_type, shape_ref->second.dims, pb_tensor);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "pack content of initializer[" << edge->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

static RetCode PackGraphInput(::onnx::GraphProto* pb_graph, const ir::Graph& graph) {
    auto topo = graph.topo.get();
    auto data = graph.data.get();

    for (uint32_t i = 0; i < topo->GetInputCount(); ++i) {
        auto eid = topo->GetInput(i);
        auto edge = topo->GetEdge(eid);

        auto shape_ref = data->shapes.find(eid);
        if (shape_ref == data->shapes.end()) {
            LOG(ERROR) << "cannot find shape of input[" << edge->GetName() << "]";
            return RC_NOT_FOUND;
        }

        auto pb_input = pb_graph->add_input();
        pb_input->set_name(edge->GetName());

        auto pb_tensor_type = pb_input->mutable_type()->mutable_tensor_type();
        pb_tensor_type->set_elem_type(utils::ConvertPplDataTypeToOnnxDataType(shape_ref->second.data_type));

        auto pb_tensor_shape = pb_tensor_type->mutable_shape();
        for (uint32_t i = 0; i < shape_ref->second.dims.size(); ++i) {
            auto dim = shape_ref->second.dims[i];
            auto pb_dim = pb_tensor_shape->add_dim();
            if (dim != INVALID_DIM_VALUE) {
                pb_dim->set_dim_value(dim);
            } else {
                auto symbol_ref = data->axis_symbols.find(make_pair(eid, i));
                if (symbol_ref == data->axis_symbols.end()) {
                    LOG(ERROR) << "cannot find symbol of dim[" << i << "] of edge[" << edge->GetName() << "]";
                    return RC_NOT_FOUND;
                }
                pb_dim->set_dim_param(symbol_ref->second);
            }
        }
    }

    return RC_SUCCESS;
}

static RetCode PackGraphOutput(::onnx::GraphProto* pb_graph, const ir::Graph& graph) {
    auto topo = graph.topo.get();
    for (uint32_t i = 0; i < topo->GetOutputCount(); ++i) {
        auto eid = topo->GetOutput(i);
        auto edge = topo->GetEdge(eid);
        auto pb_output = pb_graph->add_output();
        pb_output->set_name(edge->GetName());
    }
    return RC_SUCCESS;
}

static RetCode PackGraph(::onnx::GraphProto* pb_graph, const ir::Graph& graph) {
    pb_graph->set_name(graph.topo->GetName());

    auto status = PackGraphNode(pb_graph, graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "PackGraphNode failed: " << GetRetCodeStr(status);
        return status;
    }

    status = PackGraphInitializer(pb_graph, graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "PackGraphInitializer failed: " << GetRetCodeStr(status);
        return status;
    }

    status = PackGraphInput(pb_graph, graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "PackGraphInput failed: " << GetRetCodeStr(status);
        return status;
    }

    status = PackGraphOutput(pb_graph, graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "PackGraphOutput failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode Serializer::Serialize(const string& output_file, const Model& model) {
    ::onnx::ModelProto pb_model;
    PackModelInfo(&pb_model, model.opset);
    auto status = PackGraph(pb_model.mutable_graph(), model.graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "PackGraph failed: " << GetRetCodeStr(status);
        return status;
    }

    ofstream ofs(output_file, ios_base::out | ios_base::binary | ios_base::trunc);
    if (!ofs.is_open()) {
        LOG(ERROR) << "open file[" << output_file << "] failed.";
        return RC_OTHER_ERROR;
    }

    bool ok = pb_model.SerializeToOstream(&ofs);
    if (!ok) {
        LOG(ERROR) << "serialize to file failed.";
        return RC_OTHER_ERROR;
    }
    ofs.close();

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
