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

#ifndef _ST_HPC_PPL_NN_IR_GRAPH_DATA_H_
#define _ST_HPC_PPL_NN_IR_GRAPH_DATA_H_

#include "ppl/nn/ir/shape.h"
#include "ppl/nn/ir/constant.h"
#include "ppl/nn/ir/attr.h"
#include <map>

namespace ppl { namespace nn { namespace ir {

struct GraphData final {
    std::map<edgeid_t, Constant> constants;
    std::map<edgeid_t, Shape> shapes;
    std::map<nodeid_t, std::shared_ptr<Attr>> attrs; // attrs can be shared with cpu engines
#ifdef PPLNN_ENABLE_ONNX_MODEL
    std::map<std::pair<edgeid_t, uint32_t>, std::string> axis_symbols; // for serializing to onnx
#endif
};

}}} // namespace ppl::nn::ir

#endif
