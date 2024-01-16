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

#include "ppl/common/str_utils.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/nn/models/onnx/graph_parser.h"
#include "ppl/nn/models/onnx/param_parser_manager.h"
#include "ppl/nn/models/onnx/utils.h"
#include "ppl/nn/ir/full_graph_topo.h"
#include "ppl/nn/ir/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

static RetCode ParseOneInitializerData(const ::onnx::TensorProto& pb_initializer, edgeid_t eid,
                                       const char* model_file_dir, ir::GraphData* data) {
    ir::Shape shape;
    ir::Constant constant;
    auto rc = utils::ParseTensorProto(pb_initializer, model_file_dir, &constant.data, &shape);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "ParseTensorProto failed: " << GetRetCodeStr(rc);
        return rc;
    }

    data->shapes.insert(make_pair(eid, shape));
    data->constants.emplace(eid, std::move(constant));

    return RC_SUCCESS;
}

static RetCode ParseGraphInitializers(const ::onnx::GraphProto& pb_graph, const char* model_file_dir,
                                      ir::GraphTopo* topo, ir::GraphData* data) {
    for (int i = 0; i < pb_graph.initializer_size(); ++i) {
        const ::onnx::TensorProto& pb_initializer = pb_graph.initializer(i);

        auto ret_pair = topo->AddEdge(pb_initializer.name());
        if (!ret_pair.second) {
            LOG(ERROR) << "duplicated initializer[" << pb_initializer.name() << "].";
            return RC_EXISTS;
        }
        auto edge = ret_pair.first;

        auto rc = ParseOneInitializerData(pb_initializer, edge->GetId(), model_file_dir, data);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "ParseOneInitializerData of edge [" << edge->GetName() << "] failed.";
            return rc;
        }

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

static RetCode ParseOneInputData(const ::onnx::ValueInfoProto& pb_input, const ir::Edge* edge, ir::GraphData* data,
                                 map<pair<edgeid_t, uint32_t>, string>* axis_symbols) {
    auto& pb_type = pb_input.type();
    if (!pb_type.has_tensor_type()) {
        LOG(ERROR) << "unsupported type[" << (int)pb_type.value_case() << "] of input[" << edge->GetName()
                   << "]. only tensor type is supported currently.";
        return RC_UNSUPPORTED;
    }

    auto& pb_tensor_type = pb_type.tensor_type();
    ir::Shape shape;
    shape.data_type = utils::ConvertOnnxDataTypeToPplDataType(pb_tensor_type.elem_type());
    shape.data_format = DATAFORMAT_NDARRAY; // default data format in onnx

    auto& pb_tensor_shape = pb_tensor_type.shape();
    for (int j = 0; j < pb_tensor_shape.dim_size(); ++j) {
        const ::onnx::TensorShapeProto::Dimension& pb_dimension = pb_tensor_shape.dim(j);
        if (pb_dimension.value_case() == ::onnx::TensorShapeProto_Dimension::kDimValue) {
            auto dim_value = pb_dimension.dim_value();
            shape.dims.push_back(dim_value);
        } else if (pb_dimension.value_case() == ::onnx::TensorShapeProto_Dimension::kDimParam) {
            shape.dims.push_back(INVALID_DIM_VALUE);
            if (axis_symbols) {
                axis_symbols->insert(make_pair(make_pair(edge->GetId(), j), pb_dimension.dim_param()));
            }
        } else {
            LOG(ERROR) << "tensor[" << pb_input.name() << "] dim is not set";
            return RC_NOT_FOUND;
        }
    }

    data->shapes.insert(make_pair(edge->GetId(), shape));

    return RC_SUCCESS;
}

static RetCode ParseGraphInputs(const ::onnx::GraphProto& pb_graph, ir::GraphTopo* topo, ir::GraphData* data,
                                map<pair<edgeid_t, uint32_t>, string>* axis_symbols) {
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

        auto rc = ParseOneInputData(pb_input, edge, data, axis_symbols);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "ParseOneInputData of [" << pb_input.name() << "] failed.";
            return rc;
        }

        topo->MarkAsInput(edge->GetId());
    }

    return RC_SUCCESS;
}

static inline string GenNodeName(uint32_t anonymous_node_count) {
    return "ppl_anonymous_node_" + ToString(anonymous_node_count);
}

static ir::Node* ParseOneNodeTopo(const ::onnx::NodeProto& pb_node, const map<string, uint64_t>& op_set,
                                  ir::GraphTopo* topo, uint32_t* anonymous_node_count) {
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
        return nullptr;
    }
    auto node = ret_pair.first;

    auto op_set_ref = op_set.find(pb_node.domain());
    if (op_set_ref == op_set.end()) {
        LOG(ERROR) << "domain[" << pb_node.domain() << "] is not imported.";
        return nullptr;
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
        edge->AddConsumer(node->GetId());
        node->AddInput(edge->GetId());
    }

    for (int i = 0; i < pb_node.output_size(); ++i) {
        const string& output_name = pb_node.output(i);

        auto ret_pair = topo->AddEdge(output_name);
        auto edge = ret_pair.first;
        if (!ret_pair.second) {
            if (edge->GetProducer() != INVALID_NODEID) {
                LOG(ERROR) << "output edge[" << output_name << "] of node[" << node_name << "] already exists.";
                return nullptr;
            }
        }

        edge->SetProducer(node->GetId());
        node->AddOutput(edge->GetId());
    }

    return node;
}

static RetCode ParseOneNodeData(const ::onnx::NodeProto& pb_node, ir::Node* node, const ParamParserExtraArgs& args,
                                ir::GraphData* data) {
    auto& node_type = node->GetType();
    ppl::nn::utils::VersionRange supported_versions;
    auto parser_info = ParamParserManager::GetInstance()->Find(node_type.domain, node_type.name, node_type.version,
                                                               &supported_versions);
    if (!parser_info) {
        if (supported_versions.first == 0 && supported_versions.last == 0) {
            LOG(ERROR) << "unsupported op: domain[" << node_type.domain << "], type[" << node_type.name << "].";
        } else {
            LOG(ERROR) << "unsupported version [" << node_type.version << "] of op: domain[" << node_type.domain
                       << "], type[" << node_type.name << "]. latest supported version(s): ["
                       << supported_versions.first << ", " << supported_versions.last << "].";
        }
        return RC_UNSUPPORTED;
    }

    if (parser_info->parse_param) {
        shared_ptr<ir::Attr> param;
        if (parser_info->create_param) {
            param = parser_info->create_param();
            if (!param) {
                LOG(ERROR) << "create param failed for op[" << node->GetName() << "]";
                return RC_OTHER_ERROR;
            }
        }

        auto status = parser_info->parse_param(pb_node, args, node, param.get());
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "parse attr of node[" << node->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }

        if (param) {
            data->attrs.emplace(node->GetId(), std::move(param));
        }
    }

    return RC_SUCCESS;
}

static RetCode ParseOneGraphNode(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args,
                                 ir::GraphData* data, uint32_t* anonymous_node_count) {
    auto node = ParseOneNodeTopo(pb_node, *args.op_set, args.topo, anonymous_node_count);
    if (!node) {
        LOG(ERROR) << "ParseOneNodeTopo failed.";
        return RC_INTERNAL_ERROR;
    }

    auto rc = ParseOneNodeData(pb_node, node, args, data);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "ParseOneNodeData of node [" << pb_node.name() << "] failed.";
        return rc;
    }

    return RC_SUCCESS;
}

static RetCode ParseGraphNodes(const ::onnx::GraphProto& pb_graph, const map<string, uint64_t>& op_set,
                               const char* model_file_dir, ir::GraphTopo* topo, ir::GraphData* data,
                               uint32_t* anonymous_node_count) {
    ParamParserExtraArgs args;
    args.op_set = &op_set;
    args.model_file_dir = model_file_dir;
    args.topo = topo;
    args.data = data;

    for (int i = 0; i < pb_graph.node_size(); ++i) {
        auto& pb_node = pb_graph.node(i);
        auto status = ParseOneGraphNode(pb_node, args, data, anonymous_node_count);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "ParseOneGraphNode for node [" << pb_node.name() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

static RetCode ParseGraphOutputs(const ::onnx::GraphProto& pb_graph, ir::GraphTopo* topo, ir::GraphData* data) {
    for (int i = 0; i < pb_graph.output_size(); ++i) {
        const ::onnx::ValueInfoProto& pb_output = pb_graph.output(i);

        auto edge = topo->GetEdge(pb_output.name());
        if (!edge) {
            LOG(ERROR) << "cannot find output[" << pb_output.name() << "] in edges.";
            return RC_NOT_FOUND;
        }

        topo->MarkAsOutput(edge->GetId());

        auto& pb_output_type = pb_output.type();
        if (pb_output_type.has_tensor_type()) {
            ir::Shape shape;

            auto& pb_tensor_type = pb_output_type.tensor_type();
            shape.data_type = utils::ConvertOnnxDataTypeToPplDataType(pb_tensor_type.elem_type());
            shape.data_format = DATAFORMAT_NDARRAY; // default data format in onnx

            auto& pb_tensor_shape = pb_tensor_type.shape();
            for (int j = 0; j < pb_tensor_shape.dim_size(); ++j) {
                auto dim_value = pb_tensor_shape.dim(j).dim_value();
                shape.dims.push_back(dim_value);
            }

            data->shapes.insert(make_pair(edge->GetId(), shape));
        }
    }

    return RC_SUCCESS;
}

static void ParseGraphExtraInput(ir::GraphTopo* topo) {
    set<edgeid_t> no_producer_edges;
    for (uint32_t i = 0; i < topo->GetConstantCount(); ++i) {
        no_producer_edges.insert(topo->GetConstant(i));
    }
    for (uint32_t i = 0; i < topo->GetInputCount(); ++i) {
        no_producer_edges.insert(topo->GetInput(i));
    }

    for (auto it = topo->CreateEdgeIter(); it->IsValid(); it->Forward()) {
        auto edge = it->Get();
        if (edge->GetProducer() == INVALID_NODEID) {
            auto ref = no_producer_edges.find(edge->GetId());
            if (ref == no_producer_edges.end()) {
                topo->MarkAsExtraInput(edge->GetId());
            }
        }
    }
}

RetCode GraphParser::Parse(const ::onnx::GraphProto& pb_graph, const map<string, uint64_t>& op_set,
                           const char* model_file_dir, ir::Graph* graph,
                           map<pair<edgeid_t, uint32_t>, string>* axis_symbols) {
    graph->topo = make_shared<ir::FullGraphTopo>();
    graph->data = make_shared<ir::GraphData>();

    auto topo = graph->topo.get();
    auto data = graph->data.get();

    topo->SetName(pb_graph.name());

    auto status = ParseGraphInitializers(pb_graph, model_file_dir, topo, data);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ParseGraphInitializers failed.";
        return status;
    }

    status = ParseGraphInputs(pb_graph, topo, data, axis_symbols);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ParseGraphInputs failed.";
        return status;
    }

    status = ParseGraphNodes(pb_graph, op_set, model_file_dir, topo, data, &anonymous_node_count_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ParseGraphNodes failed.";
        return status;
    }

    status = ParseGraphOutputs(pb_graph, topo, data);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ParseGraphOutputs failed.";
        return status;
    }

    ParseGraphExtraInput(topo);

    return RC_SUCCESS;
}

/* ------------------------------------------------------------------------- */

struct ConstantIndexInfo final {
    ConstantIndexInfo() {}
    ConstantIndexInfo(edgeid_t e, int p) : eid(e), pb_idx(p) {}
    edgeid_t eid; // id in GraphTopo
    int pb_idx; // index in onnx::GraphProto::initializer
};

static RetCode CreateAllInitializerAndInputEdges(const ::onnx::GraphProto& pb_graph,
                                                 map<string, ConstantIndexInfo>* constants, ir::GraphTopo* topo) {
    for (int i = 0; i < pb_graph.initializer_size(); ++i) {
        const ::onnx::TensorProto& pb_initializer = pb_graph.initializer(i);
        auto ret_pair = topo->AddEdge(pb_initializer.name());
        if (!ret_pair.second) {
            LOG(ERROR) << "duplicated edge [" << pb_initializer.name() << "].";
            return RC_EXISTS;
        }
        constants->insert(make_pair(pb_initializer.name(), ConstantIndexInfo(ret_pair.first->GetId(), i)));
    }

    for (int i = 0; i < pb_graph.input_size(); ++i) {
        auto& pb_input = pb_graph.input(i);
        if (constants->find(pb_input.name()) != constants->end()) {
            continue;
        }
        auto ret_pair = topo->AddEdge(pb_input.name());
        if (!ret_pair.second) {
            LOG(ERROR) << "duplicated input [" << pb_input.name() << "].";
            return RC_EXISTS;
        }
    }

    return RC_SUCCESS;
}

static RetCode CreateAllNodesAndEdges(const ::onnx::GraphProto& pb_graph, const map<string, uint64_t>& op_set,
                                      ir::GraphTopo* topo, uint32_t* anonymous_node_count) {
    for (int i = 0; i < pb_graph.node_size(); ++i) {
        auto& pb_node = pb_graph.node(i);
        auto node = ParseOneNodeTopo(pb_node, op_set, topo, anonymous_node_count);
        if (!node) {
            LOG(ERROR) << "ParseOneNodeTopo failed.";
            return RC_INTERNAL_ERROR;
        }
    }
    return RC_SUCCESS;
}

static RetCode CollectNodesAndEdges(const char** inputs, uint32_t nr_input, const char** outputs, uint32_t nr_output,
                                    const map<string, ConstantIndexInfo>& constants, ir::GraphTopo* topo,
                                    set<nodeid_t>* nids, set<edgeid_t>* eids) {
    set<nodeid_t> begin_nodes;
    for (uint32_t i = 0; i < nr_input; ++i) {
        const char* input_name = inputs[i];
        auto edge = topo->GetEdge(input_name);
        if (!edge) {
            LOG(ERROR) << "cannot find input [" << input_name << "]";
            return RC_INVALID_VALUE;
        }

        for (auto it = edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
            begin_nodes.insert(it.Get());
        }

        edge->SetProducer(INVALID_NODEID);

        if (constants.find(input_name) == constants.end()) {
            topo->MarkAsInput(edge->GetId());
        }
    }

    set<nodeid_t> end_nodes;
    for (uint32_t i = 0; i < nr_output; ++i) {
        const char* output_name = outputs[i];
        auto edge = topo->GetEdge(output_name);
        if (!edge) {
            LOG(ERROR) << "cannot find output [" << output_name << "]";
            return RC_INVALID_VALUE;
        }

        auto producer_id = edge->GetProducer();
        if (producer_id != INVALID_NODEID) {
            end_nodes.insert(producer_id);
        }

        edge->ClearConsumer();
        topo->MarkAsOutput(edge->GetId());
    }

    ppl::nn::utils::ReversedDfs(
        topo->GetCurrentNodeIdBound(),
        [&end_nodes](const function<void(nodeid_t)>& f) -> void {
            for (auto it = end_nodes.begin(); it != end_nodes.end(); ++it) {
                f(*it);
            }
        },
        [topo](nodeid_t nid, const function<void(nodeid_t)>& f) -> void {
            auto prevs = topo->FindPredecessors(nid);
            for (auto p = prevs.begin(); p != prevs.end(); ++p) {
                f(*p);
            }
        },
        [topo, nids, eids](nodeid_t nid) -> void {
            nids->insert(nid);
            auto node = topo->GetNode(nid);
            for (uint32_t i = 0; i < node->GetInputCount(); ++i) {
                eids->insert(node->GetInput(i));
            }
            for (uint32_t i = 0; i < node->GetExtraInputCount(); ++i) {
                eids->insert(node->GetExtraInput(i));
            }
            for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
                eids->insert(node->GetOutput(i));
            }
        },
        [&begin_nodes](nodeid_t nid) -> bool {
            return (begin_nodes.find(nid) != begin_nodes.end());
        });

    return RC_SUCCESS;
}

// remove nodes and edges that are not in `nids` and `eids` respectively
static void RemoveUnusedNodesAndEdges(const set<nodeid_t>& nids, const set<edgeid_t>& eids, ir::GraphTopo* topo) {
    for (uint32_t i = 0; i < topo->GetCurrentNodeIdBound(); ++i) {
        if (nids.find(i) == nids.end()) {
            topo->DelNode(i);
        }
    }
    for (uint32_t i = 0; i < topo->GetCurrentEdgeIdBound(); ++i) {
        if (eids.find(i) == eids.end()) {
            topo->DelEdge(i);
        }
    }
}

static RetCode ParseInputInfo(const ::onnx::GraphProto& pb_graph, ir::GraphTopo* topo, ir::GraphData* data,
                              map<pair<edgeid_t, uint32_t>, string>* axis_symbols) {
    for (int i = 0; i < pb_graph.input_size(); ++i) {
        const ::onnx::ValueInfoProto& pb_input = pb_graph.input(i);
        auto eid = topo->GetInput(pb_input.name());
        if (eid != INVALID_EDGEID) {
            auto edge = topo->GetEdge(eid);
            auto rc = ParseOneInputData(pb_input, edge, data, axis_symbols);
            if (rc != RC_SUCCESS) {
                LOG(ERROR) << "ParseOneInputData of [" << edge->GetName() << "] failed.";
                return rc;
            }
        }
    }

    return RC_SUCCESS;
}

static RetCode ParseConstantInfo(const ::onnx::GraphProto& pb_graph, const map<string, ConstantIndexInfo>& constants,
                                 const set<edgeid_t>& eids, const char* model_file_dir, ir::GraphTopo* topo,
                                 ir::GraphData* data) {
    for (auto c = constants.begin(); c != constants.end(); ++c) {
        auto eid = c->second.eid;
        if (eids.find(eid) != eids.end()) {
            auto rc = ParseOneInitializerData(pb_graph.initializer(c->second.pb_idx), eid, model_file_dir, data);
            if (rc != RC_SUCCESS) {
                auto edge = topo->GetEdge(eid);
                LOG(ERROR) << "ParseOneInitializerData of [" << edge->GetName() << "] failed.";
                return rc;
            }
            topo->MarkAsConstant(eid);
        }
    }

    return RC_SUCCESS;
}

static RetCode ParseNodeInfo(const ::onnx::GraphProto& pb_graph, const map<string, uint64_t>& op_set,
                             const char* model_file_dir, ir::GraphTopo* topo, ir::GraphData* data) {
    ParamParserExtraArgs args;
    args.op_set = &op_set;
    args.model_file_dir = model_file_dir;
    args.topo = topo;
    args.data = data;

    for (int i = 0; i < pb_graph.node_size(); ++i) {
        auto& pb_node = pb_graph.node(i);
        auto node = topo->GetNode(pb_node.name());
        if (!node) {
            continue;
        }

        auto rc = ParseOneNodeData(pb_node, node, args, data);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "ParseOneNodeData failed: " << GetRetCodeStr(rc);
            return rc;
        }
    }

    return RC_SUCCESS;
}

RetCode GraphParser::Parse(const ::onnx::GraphProto& pb_graph, const map<string, uint64_t>& op_set,
                           const char* model_file_dir, const char** inputs, uint32_t nr_input, const char** outputs,
                           uint32_t nr_output, ir::Graph* graph, map<pair<edgeid_t, uint32_t>, string>* axis_symbols) {
    graph->topo = make_shared<ir::FullGraphTopo>();
    graph->data = make_shared<ir::GraphData>();

    auto topo = graph->topo.get();
    auto data = graph->data.get();

    topo->SetName(pb_graph.name());

    map<string, ConstantIndexInfo> constants;
    auto rc = CreateAllInitializerAndInputEdges(pb_graph, &constants, topo);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "CreateAllInitializerAndInputEdges failed.";
        return rc;
    }

    rc = CreateAllNodesAndEdges(pb_graph, op_set, topo, &anonymous_node_count_);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "CreateAllNodesAndEdges failed.";
        return rc;
    }

    set<nodeid_t> nids, eids;
    rc = CollectNodesAndEdges(inputs, nr_input, outputs, nr_output, constants, topo, &nids, &eids);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "CollectNodesAndEdges failed.";
        return rc;
    }

    RemoveUnusedNodesAndEdges(nids, eids, topo);

    rc = ParseInputInfo(pb_graph, topo, data, axis_symbols);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "ParseInputInfo failed.";
        return rc;
    }

    rc = ParseConstantInfo(pb_graph, constants, eids, model_file_dir, topo, data);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "ParseConstantInfo failed.";
        return rc;
    }

    rc = ParseNodeInfo(pb_graph, op_set, model_file_dir, topo, data);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "ParseNodeInfo failed.";
        return rc;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
