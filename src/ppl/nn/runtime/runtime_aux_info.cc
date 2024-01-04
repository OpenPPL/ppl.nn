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

#include "ppl/nn/runtime/runtime_aux_info.h"
#include "ppl/nn/runtime/utils.h"
#include "ppl/nn/ir/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

RetCode RuntimeAuxInfo::Init(const ir::GraphTopo* topo, const set<edgeid_t>& reserved_edgeids) {
    utils::DfsDeeperFirst(topo, [this](nodeid_t nid) -> void {
        this->sorted_nodes.push_back(nid);
    });

    auto status = utils::GenEdgeLastConsumer(topo, sorted_nodes, reserved_edgeids, &edge_last_consumer);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenEdgeLastConsumer failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

}} // namespace ppl::nn
