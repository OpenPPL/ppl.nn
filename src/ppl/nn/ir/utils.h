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

#ifndef _ST_HPC_PPL_NN_IR_UTILS_H_
#define _ST_HPC_PPL_NN_IR_UTILS_H_

#include "ppl/nn/common/types.h"
#include <vector>
#include <functional>

namespace ppl { namespace nn { namespace utils {

void Dfs(nodeid_t max_node_id, const std::function<nodeid_t()>& get_next_node,
         const std::function<void(nodeid_t, const std::function<void(nodeid_t)>&)>& for_each_predecessor,
         const std::function<void(nodeid_t)>& process,
         const std::function<bool(nodeid_t, nodeid_t)>& less_than = nullptr);

void Bfs(nodeid_t max_node_id, const std::function<nodeid_t()>& get_next_node,
         const std::function<void(nodeid_t, const std::function<void(nodeid_t)>&)>& for_each_predecessor,
         const std::function<void(nodeid_t, const std::function<void(nodeid_t)>&)>& for_each_successor,
         const std::function<void(nodeid_t, uint32_t level)>& process);

}}} // namespace ppl::nn::utils

#endif
