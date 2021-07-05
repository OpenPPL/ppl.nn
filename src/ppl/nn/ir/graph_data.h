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

#include "ppl/common/types.h"
#include <string>
#include <vector>
#include <map>

namespace ppl { namespace nn { namespace ir {

struct Shape final {
    ppl::common::datatype_t data_type;
    ppl::common::dataformat_t data_format;
    std::vector<int64_t> dims;
};

struct Constant final {
    std::string data;
};

struct GraphData final {
    std::map<edgeid_t, Constant> constants;
    std::map<edgeid_t, Shape> shapes;
    std::map<nodeid_t, std::shared_ptr<void>> attrs; // attrs can be shared with cpu engines
};

}}} // namespace ppl::nn::ir

#endif
