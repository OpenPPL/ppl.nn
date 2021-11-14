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

#include "ppl/nn/common/types.h"
#include "ppl/nn/runtime/runtime_graph_info.h"
#include <vector>

namespace ppl { namespace nn {

/**
   @class RuntimeAuxInfo
   @brief auxiliary info for runtime stage
*/
struct RuntimeAuxInfo final {
    /** node ids in topological order */
    std::vector<nodeid_t> sorted_nodes;

    /** a tensor can be released right after the last consumer finish executing */
    std::vector<nodeid_t> tensor_last_consumer;
};

ppl::common::RetCode GenerateRuntimeAuxInfo(const ir::GraphTopo*, RuntimeAuxInfo*);

}} // namespace ppl::nn

#endif
