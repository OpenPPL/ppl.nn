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

#include "ppl/nn/optimizers/constant_node_optimizer.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/params/onnx/constant_param.h"
using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

namespace ppl { namespace nn {

// move constant op's output edge to graph constant & delete constant op
ppl::common::RetCode ConstantNodeOptimizer::Optimize(ir::Graph* graph) const {
    auto& attrs = graph->data->attrs;
    auto& constants = graph->data->constants;
    auto& shapes = graph->data->shapes;

    for (auto it = graph->topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        if (node->GetType().domain.empty() && node->GetType().name == "Constant") {
            auto constant_node = node;
            auto edge = graph->topo->GetEdge(node->GetOutput(0));
            if (!edge) {
                LOG(ERROR) << "cannot find constant node[" << constant_node->GetName() << "]'s output edge.";
                return RC_NOT_FOUND;
            }
            auto edge_id = edge->GetId();

            auto param_it = attrs.find(constant_node->GetId());
            if (param_it == attrs.end()) {
                LOG(ERROR) << "cannot find constant node[" << constant_node->GetName() << "]'s param.";
                graph->topo->DelEdge(edge_id);
                return RC_NOT_FOUND;
            }
            auto param = (const ConstantParam*)param_it->second.get();

            // copy constant info to graph
            auto constant_ret = constants.insert(make_pair(edge_id, ir::Constant()));
            constant_ret.first->second.data = param->data;

            graph->topo->MarkAsConstant(edge_id);

            auto shape_ret = shapes.insert(make_pair(edge_id, ir::Shape()));
            shape_ret.first->second.data_type = param->data_type;
            shape_ret.first->second.data_format = DATAFORMAT_NDARRAY;
            shape_ret.first->second.dims = param->dims;

            // delete constant node
            edge->SetProducer(INVALID_NODEID); // clear producer
            graph->topo->DelNode(constant_node->GetId());
        }
    }

    return RC_SUCCESS;
}

}} // namespace ppl::nn
