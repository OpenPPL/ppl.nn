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

#include "ppl/nn/optimizers/engine_graph_partitioner.h"
#include "ppl/nn/common/logger.h"
#include <set>
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

static void DoPartition(const vector<nodeid_t>& nodes, const ir::GraphTopo* topo, EngineImpl* engine,
                        vector<pair<EngineImpl*, vector<nodeid_t>>>* partitions) {
    set<nodeid_t> nodes_left;
    nodes_left.insert(nodes.begin(), nodes.end());

    do {
        vector<nodeid_t> nodes_stack;
        nodes_stack.reserve(nodes_left.size());

        vector<nodeid_t> connected_nodes;
        connected_nodes.reserve(nodes_left.size());

        auto nodeset_iter = nodes_left.begin();
        nodes_stack.push_back(*nodeset_iter);
        nodes_left.erase(nodeset_iter);

        do {
            auto nid = nodes_stack.back();
            nodes_stack.pop_back();
            connected_nodes.push_back(nid);

            auto prev_ids = topo->FindPredecessors(nid);
            for (auto it = prev_ids.begin(); it != prev_ids.end(); ++it) {
                if (nodes_left.erase(*it) > 0) {
                    nodes_stack.push_back(*it);
                }
            }

            auto next_ids = topo->FindSuccessors(nid);
            for (auto it = next_ids.begin(); it != next_ids.end(); ++it) {
                if (nodes_left.erase(*it) > 0) {
                    nodes_stack.push_back(*it);
                }
            }
        } while (!nodes_stack.empty());

        partitions->emplace_back(engine, std::move(connected_nodes));
    } while (!nodes_left.empty());
}

static EngineImpl* FindEngine(utils::SharedResource* resource, const ir::Node* node) {
    auto engines = &resource->engines;
    for (auto it = engines->begin(); it != engines->end(); ++it) {
        auto engine = *it;
        if (engine->Supports(node)) {
            return engine;
        }
    }

    return nullptr;
}

RetCode EngineGraphPartitioner::Partition(utils::SharedResource* resource, ir::GraphTopo* topo,
                                          vector<pair<EngineImpl*, vector<nodeid_t>>>* partitions) const {
    map<EngineImpl*, vector<nodeid_t>> engine_partitions;
    for (auto iter = topo->CreateNodeIter(); iter->IsValid(); iter->Forward()) {
        auto node = iter->Get();
        auto engine = FindEngine(resource, node);
        if (!engine) {
            const ir::Node::Type& type = node->GetType();
            LOG(ERROR) << "cannot find implementation of op: domain[" << type.domain << "], type[" << type.name
                       << "], version[" << type.version << "]";
            return RC_UNSUPPORTED;
        }

        auto ret_pair = engine_partitions.insert(make_pair(engine, vector<nodeid_t>()));
        ret_pair.first->second.push_back(node->GetId());
    }

    if (engine_partitions.size() == 1) {
        auto ref = engine_partitions.begin();
        partitions->emplace_back(ref->first, std::move(ref->second));
    } else {
        for (auto it = engine_partitions.begin(); it != engine_partitions.end(); ++it) {
            DoPartition(it->second, topo, it->first, partitions);
        }
    }

    LOG(INFO) << "total partition(s) of graph[" << topo->GetName() << "]: " << partitions->size() << ".";

    return RC_SUCCESS;
}

}} // namespace ppl::nn
