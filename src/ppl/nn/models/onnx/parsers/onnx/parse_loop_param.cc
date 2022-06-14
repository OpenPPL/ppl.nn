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

#include "ppl/nn/models/onnx/parsers/onnx/parse_loop_param.h"
#include "ppl/nn/models/onnx/graph_parser.h"
#include "ppl/nn/models/onnx/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseLoopParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node* node,
                       ir::Attr* arg) {
    auto param = static_cast<LoopParam*>(arg);
    if (pb_node.attribute_size() != 1) {
        LOG(ERROR) << "invalid attribute size[" << pb_node.attribute_size() << "] != 1.";
        return RC_INVALID_VALUE;
    }
    auto& attr = pb_node.attribute(0);
    if (attr.type() != ::onnx::AttributeProto_AttributeType_GRAPH) {
        LOG(ERROR) << "unsupported attribute type[" << ::onnx::AttributeProto_AttributeType_Name(attr.type()) << "]";
        return RC_INVALID_VALUE;
    }

    GraphParser parser;
    auto status = parser.Parse(attr.g(), *args.op_set, args.model_file_dir, &(param->graph));
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "parse subgraph of loop pb_node[" << pb_node.name() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    utils::ResolveExtraInputs(param->graph.topo.get(), node, args.topo);

    return RC_SUCCESS;
}

RetCode PackLoopParam(const ir::Node*, const ir::Attr*, ::onnx::NodeProto*) {
    return RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::onnx
