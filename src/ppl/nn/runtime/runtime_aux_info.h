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

#ifndef _ST_HPC_PPL_NN_RUNTIME_RUNTIME_AUX_INFO_H_
#define _ST_HPC_PPL_NN_RUNTIME_RUNTIME_AUX_INFO_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/common/types.h"
#include "ppl/nn/ir/graph_topo.h"
#include <set>
#include <vector>

namespace ppl { namespace nn {

/**
   @class RuntimeAuxInfo
   @brief auxiliary info for runtime stage
*/
struct RuntimeAuxInfo final {
    ppl::common::RetCode Init(const ir::GraphTopo*, const std::set<edgeid_t>&);

    /** node ids in topological order */
    std::vector<nodeid_t> sorted_nodes;

    /** an `EdgeObject` can be released right after the last consumer finish executing in `sorted_nodes` */
    std::vector<nodeid_t> edge_last_consumer;
};

}} // namespace ppl::nn

#endif
