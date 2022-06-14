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

#include "ppl/nn/models/utils.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_pad_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ParsePadParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node* node,
                      ir::Attr* arg) {
    auto param = static_cast<PadParam*>(arg);

    string mode;
    utils::GetNodeAttr(pb_node, "mode", &mode, "constant");
    if (mode == "constant") {
        param->mode = PadParam::PAD_MODE_CONSTANT;
    } else if (mode == "reflect") {
        param->mode = PadParam::PAD_MODE_REFLECT;
    } else if (mode == "edge") {
        param->mode = PadParam::PAD_MODE_EDGE;
    } else {
        LOG(ERROR) << "Invalid pad mode " << mode << ".";
        return RC_INVALID_VALUE;
    }

    auto& node_type = node->GetType();
    if (node_type.version >= 2 && node_type.version < 11) {
        vector<int64_t> pads;
        utils::GetNodeAttr(pb_node, "pads", &pads);

        auto new_edge_name = node->GetName() + "_pad_pads_" + std::to_string(args.topo->GetCurrentEdgeIdBound());
        auto edge = ppl::nn::utils::Add1DInitializer(args.topo, args.data, new_edge_name, pads, DATATYPE_INT64);
        if (!edge) {
            LOG(ERROR) << "add initializer[" << new_edge_name << "] failed.";
            return RC_OTHER_ERROR;
        }
        node->AddInput(edge->GetId());

        float value;
        utils::GetNodeAttr(pb_node, "value", &value, 0.0);

        new_edge_name = node->GetName() + "_pad_value_" + std::to_string(args.topo->GetCurrentEdgeIdBound());
        edge = ppl::nn::utils::AddScalarInitializer(args.topo, args.data, new_edge_name, value, DATATYPE_FLOAT32);
        if (!edge) {
            LOG(ERROR) << "add initializer[" << new_edge_name << "] failed.";
            return RC_OTHER_ERROR;
        }
        node->AddInput(edge->GetId());

        node_type.version = 11;
    }
    return RC_SUCCESS;
}

RetCode PackPadParam(const ir::Node*, const ir::Attr* arg, ::onnx::NodeProto* pb_node) {
    auto param = static_cast<const PadParam*>(arg);

    if (param->mode == PadParam::PAD_MODE_CONSTANT) {
        utils::SetNodeAttr(pb_node, "mode", "constant");
    } else if (param->mode == PadParam::PAD_MODE_REFLECT) {
        utils::SetNodeAttr(pb_node, "mode", "reflect");
    } else if (param->mode == PadParam::PAD_MODE_EDGE) {
        utils::SetNodeAttr(pb_node, "mode", "edge");
    } else {
        LOG(ERROR) << "invalid pad mode[" << param->mode << "]";
        return RC_INVALID_VALUE;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
