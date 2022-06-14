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
#include "ppl/nn/models/onnx/parsers/onnx/parse_topk_param.h"
#include "ppl/nn/models/onnx/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseTopKParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node* node,
                       ir::Attr* arg) {
    auto param = static_cast<TopKParam*>(arg);
    utils::GetNodeAttr(pb_node, "axis", &param->axis, -1);
    utils::GetNodeAttr(pb_node, "largest", &param->largest, 1);
    utils::GetNodeAttr(pb_node, "sorted", &param->sorted, 1);

    auto& node_type = node->GetType();
    if (node_type.version < 10) {
        int64_t k;
        utils::GetNodeAttr(pb_node, "k", &k, -1);
        auto new_edge_name = node->GetName() + "_topk_k_" + std::to_string(args.topo->GetCurrentEdgeIdBound());
        auto edge = ppl::nn::utils::AddScalarInitializer(args.topo, args.data, new_edge_name, k, DATATYPE_INT64);
        if (!edge) {
            LOG(ERROR) << "add initializer[" << new_edge_name << "] failed.";
            return RC_OTHER_ERROR;
        }
        node->AddInput(edge->GetId());
        node_type.version = 11;
    }

    return RC_SUCCESS;
}

RetCode PackTopKParam(const ir::Node*, const ir::Attr* arg, ::onnx::NodeProto* pb_node) {
    auto param = static_cast<const TopKParam*>(arg);
    utils::SetNodeAttr(pb_node, "axis", param->axis);
    utils::SetNodeAttr(pb_node, "largest", param->largest);
    utils::SetNodeAttr(pb_node, "sorted", param->sorted);
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
