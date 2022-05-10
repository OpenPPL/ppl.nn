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

#include "ppl/nn/runtime/runtime_init_info.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

static void InitValidNodeFlags(const ir::GraphTopo* topo, vector<bool>* flags) {
    flags->resize(topo->GetCurrentNodeIdBound(), false);
    for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        flags->at(it->Get()->GetId()) = true;
    }
}

static void InitValidEdgeFlags(const ir::GraphTopo* topo, vector<bool>* flags) {
    flags->resize(topo->GetCurrentEdgeIdBound(), false);
    for (auto it = topo->CreateEdgeIter(); it->IsValid(); it->Forward()) {
        flags->at(it->Get()->GetId()) = true;
    }
}

static void InitName2Nodeid(const ir::GraphTopo* topo, map<string, nodeid_t>* name2nodeid) {
    for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        name2nodeid->insert(make_pair(node->GetName(), node->GetId()));
    }
}

RetCode RuntimeInitInfo::Init(const ir::GraphTopo* topo) {
    InitValidNodeFlags(topo, &valid_node_flags);
    InitValidEdgeFlags(topo, &valid_edge_flags);
    InitName2Nodeid(topo, &name2nodeid);
    return RC_SUCCESS;
}

}}
