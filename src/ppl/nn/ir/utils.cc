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

#include "ppl/nn/ir/utils.h"
#include <queue>
#include <algorithm>
using namespace std;

namespace ppl { namespace nn { namespace utils {

struct DfsNodeInfo final {
    DfsNodeInfo(nodeid_t nid = INVALID_NODEID, bool r = false) : id(nid), resolved(r) {}
    nodeid_t id;
    bool resolved;
};

void ReversedDfs(nodeid_t max_node_id, const function<void(const function<void(nodeid_t)>&)>& for_each_end_node,
                 const function<void(nodeid_t, const function<void(nodeid_t)>&)>& for_each_predecessor,
                 const function<void(nodeid_t)>& process_in_order, const function<bool(nodeid_t)>& stop,
                 const function<bool(nodeid_t, nodeid_t)>& less_than) {
    vector<DfsNodeInfo> node_stack;
    node_stack.reserve(max_node_id);
    for_each_end_node([&node_stack](nodeid_t nid) -> void {
        node_stack.emplace_back(nid, false);
    });

    vector<bool> visited(max_node_id, false);

    while (!node_stack.empty()) {
        auto item = node_stack.back();
        node_stack.pop_back();

        if (item.resolved) {
            process_in_order(item.id);
            continue;
        }

        if (visited[item.id]) {
            continue;
        }

        visited[item.id] = true;
        node_stack.emplace_back(item.id, true);

        if (stop && stop(item.id)) {
            continue;
        }

        if (less_than) {
            vector<nodeid_t> prev_ids;
            for_each_predecessor(item.id, [&visited, &prev_ids](nodeid_t prev) -> void {
                if (!visited[prev]) {
                    prev_ids.push_back(prev);
                }
            });
            if (!prev_ids.empty()) {
                std::sort(prev_ids.begin(), prev_ids.end(), less_than);
                for (auto x = prev_ids.begin(); x != prev_ids.end(); ++x) {
                    node_stack.emplace_back(*x, false);
                }
            }
        } else {
            for_each_predecessor(item.id, [&visited, &node_stack](nodeid_t prev) -> void {
                if (!visited[prev]) {
                    node_stack.emplace_back(prev, false);
                }
            });
        }
    }
}

struct BfsNodeInfo final {
    BfsNodeInfo(nodeid_t nid = INVALID_NODEID, uint32_t l = 0) : id(nid), level(l) {}
    nodeid_t id;
    uint32_t level;
};

void Bfs(nodeid_t max_node_id, const function<void(const function<void(nodeid_t)>&)>& for_each_node,
         const function<uint32_t(nodeid_t)>& get_predecessor_count,
         const function<void(nodeid_t, const function<void(nodeid_t)>&)>& for_each_successor,
         const function<void(nodeid_t, uint32_t)>& process) {
    queue<BfsNodeInfo> q;
    vector<nodeid_t> refcount(max_node_id, 0);

    for_each_node([&refcount, &get_predecessor_count, &q](nodeid_t nid) -> void {
        refcount[nid] = get_predecessor_count(nid);
        if (refcount[nid] == 0) {
            q.push(BfsNodeInfo(nid, 0));
        }
    });

    while (!q.empty()) {
        auto item = q.front();
        q.pop();
        process(item.id, item.level);

        const uint32_t next_level = item.level + 1;
        for_each_successor(item.id, [next_level, &refcount, &q](nodeid_t next) -> void {
            --refcount[next];
            if (refcount[next] == 0) {
                q.push(BfsNodeInfo(next, next_level));
            }
        });
    }
}

void DfsDeeperFirst(const ir::GraphTopo* topo, const function<void(nodeid_t)>& process) {
    // get node levels for second stage
    vector<uint32_t> nid2level(topo->GetCurrentNodeIdBound(), 0);
    Bfs(
        topo->GetCurrentNodeIdBound(),
        [topo](const function<void(nodeid_t)>& f) -> void {
            for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
                f(it->Get()->GetId());
            }
        },
        [topo](nodeid_t nid) -> uint32_t {
            return topo->FindPredecessors(nid).size();
        },
        [topo](nodeid_t nid, const function<void(nodeid_t)>& f) -> void {
            auto nexts = topo->FindSuccessors(nid);
            for (auto x : nexts) {
                f(x);
            }
        },
        [&nid2level](nodeid_t nid, uint32_t level) -> void {
            nid2level[nid] = level;
        });

    ReversedDfs(
        topo->GetCurrentNodeIdBound(),
        [topo](const function<void(nodeid_t)>& f) -> void {
            auto leaf_nodes = topo->FindLeafNodes();
            for (auto x = leaf_nodes.begin(); x != leaf_nodes.end(); ++x) {
                f(*x);
            }
        },
        [topo](nodeid_t nid, const function<void(nodeid_t)>& f) -> void {
            auto prevs = topo->FindPredecessors(nid);
            for (auto x : prevs) {
                f(x);
            }
        },
        process, {},
        [&nid2level](nodeid_t a, nodeid_t b) -> bool {
            // nodes in the longer path will be evaluated first
            return (nid2level[a] < nid2level[b]);
        });
}

}}} // namespace ppl::nn::utils
