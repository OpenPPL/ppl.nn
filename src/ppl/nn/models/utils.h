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

#ifndef _ST_HPC_PPL_NN_MODELS_UTILS_H
#define _ST_HPC_PPL_NN_MODELS_UTILS_H

#include "ppl/nn/ir/graph.h"
#include "ppl/common/types.h"

namespace ppl { namespace nn { namespace utils {

template <typename T>
ir::Edge* AddScalarInitializer(ir::GraphTopo* topo, ir::GraphData* data, const std::string& key, const T& value,
                               ppl::common::datatype_t data_type) {
    auto edge_ret = topo->AddEdge(key);
    if (!edge_ret.first) {
        return nullptr;
    }
    auto edge = edge_ret.first;
    auto eid = edge->GetId();

    topo->MarkAsConstant(eid);

    auto constant_ret = data->constants.insert(std::make_pair(eid, ir::Constant()));
    constant_ret.first->second.data.Assign((const char*)&value, sizeof(value));

    auto shape_ret = data->shapes.insert(std::make_pair(eid, ir::Shape()));
    shape_ret.first->second.data_type = data_type;
    shape_ret.first->second.data_format = ppl::common::DATAFORMAT_NDARRAY;
    shape_ret.first->second.dims.push_back(1);

    return edge;
}

template <typename T>
ir::Edge* Add1DInitializer(ir::GraphTopo* topo, ir::GraphData* data, const std::string& key,
                           const std::vector<T>& value, ppl::common::datatype_t data_type) {
    auto edge_ret = topo->AddEdge(key);
    if (!edge_ret.first) {
        return nullptr;
    }
    auto edge = edge_ret.first;
    auto eid = edge->GetId();

    topo->MarkAsConstant(eid);

    auto constant_ret = data->constants.insert(std::make_pair(eid, ir::Constant()));
    constant_ret.first->second.data.Assign((const char*)value.data(), value.size() * sizeof(T));

    auto shape_ret = data->shapes.insert(std::make_pair(eid, ir::Shape()));
    shape_ret.first->second.data_type = data_type;
    shape_ret.first->second.data_format = ppl::common::DATAFORMAT_NDARRAY;
    shape_ret.first->second.dims.push_back(value.size());

    return edge;
}

}}} // namespace ppl::nn::utils

#endif
