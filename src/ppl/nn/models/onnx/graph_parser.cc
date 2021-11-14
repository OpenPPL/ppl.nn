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

#include "ppl/nn/models/onnx/graph_parser.h"
#include "ppl/nn/models/onnx/param_parser_manager.h"
#include "ppl/nn/models/onnx/utils.h"
#include "ppl/nn/ir/full_graph_topo.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

static RetCode ParseGraphInitializer(const ::onnx::GraphProto& pb_graph, ir::GraphTopo* topo, ir::GraphData* data) {
    for (int i = 0; i < pb_graph.initializer_size(); ++i) {
        const ::onnx::TensorProto& pb_initializer = pb_graph.initializer(i);
        if (pb_initializer.external_data_size() > 0) {
            LOG(ERROR) << "external data is not supported now.";
            return RC_UNSUPPORTED;
        }

        auto ret_pair = topo->AddEdge(pb_initializer.name());
        if (!ret_pair.second) {
            LOG(ERROR) << "duplicated initializer[" << pb_initializer.name() << "].";
            return RC_EXISTS;
        }
        auto edge = ret_pair.first;

        ir::Shape shape;
        ir::Constant constant;
        auto status = utils::ParseTensorProto(pb_initializer, &constant.data, &shape);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "ParseTensorProto failed: " << GetRetCodeStr(status);
            return status;
        }

        data->shapes.insert(make_pair(edge->GetId(), shape));
        data->constants.emplace(edge->GetId(), std::move(constant));
        topo->MarkAsConstant(edge->GetId());
    }

    return RC_SUCCESS;
}

static void GetAllConstantNames(const ::onnx::GraphProto& pb_graph, set<string>* constants) {
    for (int i = 0; i < pb_graph.initializer_size(); ++i) {
        auto& pb_initializer = pb_graph.initializer(i);
        constants->insert(pb_initializer.name());
    }
}

static RetCode ParseGraphInput(const ::onnx::GraphProto& pb_graph, ir::GraphTopo* topo, ir::GraphData* data) {
    set<string> constants;
    GetAllConstantNames(pb_graph, &constants);

    for (int i = 0; i < pb_graph.input_size(); ++i) {
        const ::onnx::ValueInfoProto& pb_input = pb_graph.input(i);

        if (constants.find(pb_input.name()) != constants.end()) {
            continue;
        }

        auto ret_pair = topo->AddEdge(pb_input.name());
        if (!ret_pair.second) {
            LOG(ERROR) << "duplicated input[" << pb_input.name() << "].";
            return RC_EXISTS;
        }
        auto edge = ret_pair.first;

        auto& pb_tensor_type = pb_input.type().tensor_type();
        ir::Shape shape;
        shape.data_type = utils::ConvertOnnxDataTypeToPplDataType(pb_tensor_type.elem_type());
        shape.data_format = DATAFORMAT_NDARRAY; // default data format in onnx

        auto& pb_tensor_shape = pb_input.type().tensor_type().shape();
        for (int j = 0; j < pb_tensor_shape.dim_size(); ++j) {
            const ::onnx::TensorShapeProto::Dimension& pb_dimension = pb_tensor_shape.dim(j);
            if (pb_dimension.value_case() == ::onnx::TensorShapeProto_Dimension::kDimValue) {
                auto dim_value = pb_tensor_shape.dim(j).dim_value();
                shape.dims.push_back(dim_value);
            } else if (pb_dimension.value_case() == ::onnx::TensorShapeProto_Dimension::kDimParam) {
                // TODO check dim param values
                shape.dims.push_back(1);
            } else {
                LOG(ERROR) << "tensor[" << pb_input.name() << "] dim is not set";
                return RC_NOT_FOUND;
            }
        }

        data->shapes.insert(make_pair(edge->GetId(), shape));
        topo->MarkAsInput(edge->GetId());
    }

    return RC_SUCCESS;
}

static string GenNodeName(uint32_t anonymous_node_count) {
    char buf[64];
    auto len = sprintf(buf, "ppl_anonymous_node_%u", anonymous_node_count);
    return string(buf, len);
}

static RetCode ParseNodeInfo(const ::onnx::NodeProto& pb_node, const map<string, uint64_t>& op_sets,
                             ir::GraphTopo* topo, ir::GraphData* data, uint32_t* anonymous_node_count) {
    string node_name;
    if (pb_node.name().empty()) {
        node_name = GenNodeName(*anonymous_node_count);
        ++(*anonymous_node_count);
    } else {
        node_name = pb_node.name();
    }

    auto ret_pair = topo->AddNode(node_name);
    if (!ret_pair.second) {
        LOG(ERROR) << "node[" << node_name << "] already exists.";
        return RC_EXISTS;
    }
    auto node = ret_pair.first;

    auto op_set_ref = op_sets.find(pb_node.domain());
    if (op_set_ref == op_sets.end()) {
        LOG(ERROR) << "domain[" << pb_node.domain() << "] is not imported.";
        return RC_INVALID_VALUE;
    }

    node->SetType(ir::Node::Type(pb_node.domain(), pb_node.op_type(), op_set_ref->second));

    for (int i = 0; i < pb_node.input_size(); ++i) {
        const string& input_name = pb_node.input(i);
        if (input_name.empty()) {
            node->AddInput(INVALID_EDGEID);
            continue;
        }

        auto ret_pair = topo->AddEdge(input_name);
        auto edge = ret_pair.first;
        if (ret_pair.second) {
            topo->MarkAsExtraInput(edge->GetId());
        }

        edge->AddConsumer(node->GetId());
        node->AddInput(edge->GetId());
    }

    for (int i = 0; i < pb_node.output_size(); ++i) {
        const string& output_name = pb_node.output(i);

        auto ret_pair = topo->AddEdge(output_name);
        if (!ret_pair.second) {
            LOG(ERROR) << "output edge[" << output_name << "] of node[" << node_name << "] already exists.";
            return RC_EXISTS;
        }
        auto edge = ret_pair.first;

        edge->SetProducer(node->GetId());
        node->AddOutput(edge->GetId());
    }

    auto parser_info = ParamParserManager::Instance()->Find(pb_node.domain(), pb_node.op_type(), op_set_ref->second);
    if (!parser_info) {
        LOG(ERROR) << "unsupported op: domain[" << pb_node.domain() << "], type[" << pb_node.op_type() << "], version["
                   << op_set_ref->second << "]";
        return RC_UNSUPPORTED;
    }

    if (!parser_info->create_param) {
        return RC_SUCCESS;
    }

    auto param = VoidPtr(parser_info->create_param(), parser_info->destroy_param);
    auto status = parser_info->parse_param(pb_node, op_sets, param.get(), node, topo);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "parse attr of node[" << node_name << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    data->attrs.emplace(node->GetId(), std::move(param));

    return RC_SUCCESS;
}

static RetCode ParseGraphNode(const ::onnx::GraphProto& pb_graph, const map<string, uint64_t>& op_sets,
                              ir::GraphTopo* topo, ir::GraphData* data, uint32_t* anonymous_node_count) {
    for (int i = 0; i < pb_graph.node_size(); ++i) {
        auto& pb_node = pb_graph.node(i);
        auto status = ParseNodeInfo(pb_node, op_sets, topo, data, anonymous_node_count);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "ParseNodeInfo for node[" << pb_node.name() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

static RetCode ParseGraphOutput(const ::onnx::GraphProto& pb_graph, ir::GraphTopo* topo, ir::GraphData* data) {
    for (int i = 0; i < pb_graph.output_size(); ++i) {
        const ::onnx::ValueInfoProto& pb_output = pb_graph.output(i);

        auto edge = topo->GetEdgeByName(pb_output.name());
        if (!edge) {
            LOG(ERROR) << "cannot find output[" << pb_output.name() << "] in edges.";
            return RC_NOT_FOUND;
        }

        ir::Shape shape;

        auto& pb_output_type = pb_output.type();
        if (pb_output_type.has_tensor_type()) {
            auto& pb_tensor_type = pb_output_type.tensor_type();
            shape.data_type = utils::ConvertOnnxDataTypeToPplDataType(pb_tensor_type.elem_type());
            shape.data_format = DATAFORMAT_NDARRAY;
        } else {
            LOG(ERROR) << "unsupported type[" << pb_output_type.value_case() << "] of output[" << edge->GetName()
                       << "]";
            return RC_UNSUPPORTED;
        }

        auto& pb_tensor_shape = pb_output.type().tensor_type().shape();
        for (int j = 0; j < pb_tensor_shape.dim_size(); ++j) {
            auto dim_value = pb_tensor_shape.dim(j).dim_value();
            shape.dims.push_back(dim_value);
        }

        data->shapes.insert(make_pair(edge->GetId(), shape));
        topo->MarkAsOutput(edge->GetId());
    }

    return RC_SUCCESS;
}

RetCode GraphParser::Parse(const ::onnx::GraphProto& pb_graph, const map<string, uint64_t>& op_sets, ir::Graph* graph) {
    graph->topo = make_shared<ir::FullGraphTopo>(pb_graph.name());
    graph->data = make_shared<ir::GraphData>();

    auto topo = graph->topo.get();
    auto data = graph->data.get();

    auto status = ParseGraphInitializer(pb_graph, topo, data);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ParseGraphInitializer failed.";
        return status;
    }

    status = ParseGraphInput(pb_graph, topo, data);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ParseGraphInput failed.";
        return status;
    }

    status = ParseGraphNode(pb_graph, op_sets, topo, data, &anonymous_node_count_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ParseGraphNode failed.";
        return status;
    }

    status = ParseGraphOutput(pb_graph, topo, data);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ParseGraphOutput failed.";
        return status;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
