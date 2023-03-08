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
#include "ppl/nn/models/utils.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_squeeze_param.h"
#include "ppl/nn/models/onnx/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseSqueezeParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node* node,
                          ir::Attr* arg) {
    auto& node_type = node->GetType();

    if (node_type.version < 13) {
        auto topo = args.topo;
        auto data = args.data;

        std::vector<int64_t> axes;
        utils::GetNodeAttr(pb_node, "axes", &axes);

        auto new_edge_name = node->GetName() + "_axes_" + ToString(topo->GetCurrentEdgeIdBound());
        auto edge = ppl::nn::utils::Add1DInitializer(topo, data, new_edge_name, axes, DATATYPE_INT64);
        if (!edge) {
            LOG(ERROR) << "add initializer[" << new_edge_name << "] failed.";
            return RC_OTHER_ERROR;
        }
        node->AddInput(edge->GetId());
        edge->AddConsumer(node->GetId());

        node_type.version = 13;
    }
    return RC_SUCCESS;
}

RetCode PackSqueezeParam(const ir::Node*, const ir::Attr* arg, ::onnx::NodeProto* pb_node) {
    auto param = static_cast<const SqueezeParam*>(arg);
    utils::SetNodeAttr(pb_node, "axes", param->axes);
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
