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

#include "ppl/nn/models/onnx/parsers/onnx/parse_clip_param.h"
#include "ppl/nn/models/onnx/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

namespace ppl { namespace nn { namespace onnx {

static ir::Edge* AddNewInitializer(ir::GraphTopo* topo, ir::GraphData* data, const string& key, float value) {
    auto edge_ret = topo->AddEdge(key);
    if (!edge_ret.first) {
        LOG(ERROR) << "add new initializer[" << key << "] failed.";
        return nullptr;
    }
    auto edge = edge_ret.first;
    auto eid = edge->GetId();

    topo->MarkAsConstant(eid);

    auto constant_ret = data->constants.insert(make_pair(eid, ir::Constant()));
    constant_ret.first->second.data.assign((const char*)(&value), sizeof(value));

    auto shape_ret = data->shapes.insert(make_pair(eid, ir::Shape()));
    shape_ret.first->second.data_type = DATATYPE_FLOAT32;
    shape_ret.first->second.data_format = DATAFORMAT_NDARRAY;
    shape_ret.first->second.dims.push_back(1);

    return edge;
}

RetCode ParseClipParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node* node,
                       ir::Attr* arg) {
    auto& node_type = node->GetType();

    if (node_type.version >= 6 && node_type.version < 11) {
        auto topo = args.topo;
        auto data = args.data;

        auto min_value = utils::GetNodeAttrByKey<float>(pb_node, "min", numeric_limits<float>::lowest());
        auto max_value = utils::GetNodeAttrByKey<float>(pb_node, "max", numeric_limits<float>::max());

        auto edge = AddNewInitializer(
            topo, data, node->GetName() + "_clip_min_" + std::to_string(topo->GetCurrentEdgeIdBound()), min_value);
        if (!edge) {
            return RC_OTHER_ERROR;
        }
        node->AddInput(edge->GetId());

        edge = AddNewInitializer(
            topo, data, node->GetName() + "_clip_max_" + std::to_string(topo->GetCurrentEdgeIdBound()), max_value);
        if (!edge) {
            return RC_OTHER_ERROR;
        }
        node->AddInput(edge->GetId());

        node_type.version = 11;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
