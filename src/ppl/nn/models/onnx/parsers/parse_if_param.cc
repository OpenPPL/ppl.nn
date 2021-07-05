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

#include "ppl/nn/models/onnx/parsers/parse_if_param.h"
#include "ppl/nn/models/onnx/graph_parser.h"
#include "ppl/nn/models/onnx/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

static uint32_t FindExtraInputIndex(const string& name, const ir::Node* node, const ir::GraphTopo* topo) {
    for (uint32_t i = 0; i < node->GetExtraInputCount(); ++i) {
        auto edge = topo->GetEdgeById(node->GetExtraInput(i));
        if (edge->GetName() == name) {
            return i;
        }
    }

    return node->GetExtraInputCount();
}

/*
  utils::ProcessGraph() may change inputs' name, we record the indices of extra inputs of
  then/else branches respectively.
*/
static RetCode CollectExtraInputIndices(const ir::GraphTopo* current, const ir::Node* parent_node,
                                        const ir::GraphTopo* parent_topo, vector<uint32_t>* indices) {
    indices->reserve(current->GetExtraInputCount());
    for (uint32_t i = 0; i < current->GetExtraInputCount(); ++i) {
        auto edge = current->GetEdgeById(current->GetExtraInput(i));
        auto idx = FindExtraInputIndex(edge->GetName(), parent_node, parent_topo);
        if (idx == parent_node->GetExtraInputCount()) {
            LOG(ERROR) << "cannot find extra input[" << edge->GetName() << "] of subgraph[" << current->GetName()
                       << "] in node[" << parent_node->GetName() << "]";
            return RC_NOT_FOUND;
        }

        indices->push_back(idx);
    }

    return RC_SUCCESS;
}

RetCode ParseIfParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node* node, ir::GraphTopo* topo) {
    auto param = (ppl::nn::common::IfParam*)arg;
    for (int i = 0; i < pb_node.attribute_size(); ++i) {
        auto& attr = pb_node.attribute(i);
        if (attr.type() != ::onnx::AttributeProto_AttributeType_GRAPH) {
            LOG(ERROR) << "unsupported attribute type[" << ::onnx::AttributeProto_AttributeType_Name(attr.type())
                       << "]";
            return RC_INVALID_VALUE;
        }

        if (attr.name() == "then_branch") {
            GraphParser parser;
            auto status = parser.Parse(attr.g(), &param->then_branch);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "parse then_branch of if op[" << pb_node.name() << "] failed: " << GetRetCodeStr(status);
                return status;
            }

            utils::ResolveExtraInputs(param->then_branch.topo.get(), node, topo);
            status = CollectExtraInputIndices(param->then_branch.topo.get(), node, topo,
                                              &param->then_extra_input_indices_in_parent_node);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "CollectExtraInputIndices of then branch failed: " << GetRetCodeStr(status);
                return status;
            }
        } else if (attr.name() == "else_branch") {
            GraphParser parser;
            auto status = parser.Parse(attr.g(), &param->else_branch);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "parse else_branch of if op[" << pb_node.name() << "] failed: " << GetRetCodeStr(status);
                return status;
            }

            utils::ResolveExtraInputs(param->else_branch.topo.get(), node, topo);
            status = CollectExtraInputIndices(param->else_branch.topo.get(), node, topo,
                                              &param->else_extra_input_indices_in_parent_node);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "CollectExtraInputIndices of else branch failed: " << GetRetCodeStr(status);
                return status;
            }
        } else {
            LOG(ERROR) << "unsupported attribute[" << attr.name() << "]";
            return RC_UNSUPPORTED;
        }
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
