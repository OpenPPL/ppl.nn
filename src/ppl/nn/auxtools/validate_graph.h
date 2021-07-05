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

#ifndef _ST_HPC_PPL_NN_AUXTOOLS_VALIDATE_GRAPH_H_
#define _ST_HPC_PPL_NN_AUXTOOLS_VALIDATE_GRAPH_H_

#include "ppl/nn/common/logger.h"
#include <set>
#include <queue>
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace utils {

static inline bool FindPrev(nodeid_t prev, nodeid_t node, const ir::GraphTopo* topo) {
    auto predecessors = topo->FindPredecessors(node);
    for (auto p = predecessors.begin(); p != predecessors.end(); ++p) {
        if (prev == *p) {
            return true;
        }
    }
    return false;
}

static inline bool FindNext(nodeid_t next, nodeid_t node, const ir::GraphTopo* topo) {
    auto successors = topo->FindSuccessors(node);
    for (auto s = successors.begin(); s != successors.end(); ++s) {
        if (next == *s) {
            return true;
        }
    }
    return false;
}

static inline bool FindConsumer(nodeid_t cid, edgeid_t eid, const ir::GraphTopo* topo) {
    auto edge = topo->GetEdgeById(eid);
    for (auto it = edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
        if (it.Get() == cid) {
            return true;
        }
    }
    return false;
}

static bool DiffNodes(const set<nodeid_t>& node_dedup, const set<nodeid_t>& all_nodes, const ir::GraphTopo* topo) {
    if (node_dedup.size() != all_nodes.size()) {
        LOG(ERROR) << "node count diff: used[" << node_dedup.size() << "] != all[" << all_nodes.size() << "]";
    }

    vector<nodeid_t> diff_result_used2all;
    diff_result_used2all.resize(node_dedup.size());
    auto end_iter = std::set_difference(node_dedup.begin(), node_dedup.end(), all_nodes.begin(), all_nodes.end(),
                                        diff_result_used2all.begin());
    diff_result_used2all.resize(end_iter - diff_result_used2all.begin());
    if (!diff_result_used2all.empty()) {
        LOG(ERROR) << "diff node set used <=> all:";
        for (auto x = diff_result_used2all.begin(); x != diff_result_used2all.end(); ++x) {
            auto node = topo->GetNodeById(*x);
            LOG(ERROR) << "    " << node->GetName();
        }
    }

    vector<nodeid_t> diff_result_all2used;
    diff_result_all2used.resize(all_nodes.size());
    end_iter = std::set_difference(all_nodes.begin(), all_nodes.end(), node_dedup.begin(), node_dedup.end(),
                                   diff_result_all2used.begin());
    diff_result_all2used.resize(end_iter - diff_result_all2used.begin());
    if (!diff_result_all2used.empty()) {
        LOG(ERROR) << "diff node set all <=> used:";
        for (auto x = diff_result_all2used.begin(); x != diff_result_all2used.end(); ++x) {
            auto node = topo->GetNodeById(*x);
            LOG(ERROR) << "    " << node->GetName();
        }
    }

    return (diff_result_used2all.empty() && diff_result_all2used.empty());
}

// some edges may be used only by graph itself, e.g. `cond` of Loop
static bool DiffEdges(const set<edgeid_t>& edge_dedup, const set<edgeid_t>& all_edges, const ir::GraphTopo* topo) {
    if (edge_dedup.size() > all_edges.size()) {
        LOG(ERROR) << "edge count diff: used[" << edge_dedup.size() << "] > all[" << all_edges.size() << "]";
        return false;
    }

    vector<edgeid_t> diff_result_used2all;
    diff_result_used2all.resize(edge_dedup.size());
    auto end_iter = std::set_difference(edge_dedup.begin(), edge_dedup.end(), all_edges.begin(), all_edges.end(),
                                        diff_result_used2all.begin());
    diff_result_used2all.resize(end_iter - diff_result_used2all.begin());
    if (!diff_result_used2all.empty()) {
        LOG(ERROR) << "diff edge set used <=> all:";
        for (auto x = diff_result_used2all.begin(); x != diff_result_used2all.end(); ++x) {
            auto edge = topo->GetEdgeById(*x);
            LOG(ERROR) << "    " << edge->GetName();
        }
    }

    return diff_result_used2all.empty();
}

static bool ValidateGraphTopo(const ir::GraphTopo* topo) {
    set<nodeid_t> node_dedup;
    set<edgeid_t> edge_dedup;
    set<nodeid_t> all_nodes;

    LOG(INFO) << "validating graph ...";

    queue<nodeid_t> q;
    for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto nid = it->Get()->GetId();
        all_nodes.insert(nid);
        auto predecessors = topo->FindPredecessors(nid);
        if (predecessors.empty()) {
            q.push(nid);
        }
    }

    while (!q.empty()) {
        auto nid = q.front();
        auto node = topo->GetNodeById(nid);
        q.pop();
        node_dedup.insert(nid);

        auto successors = topo->FindSuccessors(nid);
        for (auto s = successors.begin(); s != successors.end(); ++s) {
            auto snid = *s;
            auto ret_pair = node_dedup.insert(snid);
            if (ret_pair.second) {
                q.push(snid);
            }

            if (!FindPrev(nid, snid, topo)) {
                auto next = topo->GetNodeById(snid);
                LOG(ERROR) << "cannot find node[" << node->GetName() << "] in successor[" << next->GetName()
                           << "]'s predecessor list.";
                return false;
            }
        }

        auto predecessors = topo->FindPredecessors(nid);
        for (auto p = predecessors.begin(); p != predecessors.end(); ++p) {
            auto pnid = *p;
            auto ret_pair = node_dedup.insert(pnid);
            if (ret_pair.second) {
                q.push(pnid);
            }

            if (!FindNext(nid, pnid, topo)) {
                auto prev = topo->GetNodeById(pnid);
                LOG(ERROR) << "cannot find node[" << node->GetName() << "] in predecessor[" << prev->GetName()
                           << "]'s successor list.";
                return false;
            }
        }

        for (uint32_t i = 0; i < node->GetInputCount(); ++i) {
            auto in = node->GetInput(i);
            if (in == INVALID_EDGEID) { // optional inputs
                continue;
            }
            if (!FindConsumer(nid, in, topo)) {
                auto edge = topo->GetEdgeById(in);
                LOG(ERROR) << "cannot find node[" << node->GetName() << "] in edge[" << edge->GetName()
                           << "]'s consumer list.";
                return false;
            }
            edge_dedup.insert(in);
        }

        for (uint32_t i = 0; i < node->GetExtraInputCount(); ++i) {
            auto in = node->GetExtraInput(i);
            if (in == INVALID_EDGEID) { // optional inputs
                continue;
            }
            if (!FindConsumer(nid, in, topo)) {
                auto edge = topo->GetEdgeById(in);
                LOG(ERROR) << "cannot find node[" << node->GetName() << "] in edge[" << edge->GetName()
                           << "]'s consumer list.";
                return false;
            }
            edge_dedup.insert(in);
        }

        for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
            auto out = node->GetOutput(i);
            auto edge = topo->GetEdgeById(out);
            if (edge->GetProducer() != nid) {
                LOG(ERROR) << "edge[" << edge->GetName() << "]'s producer is not [" << node->GetName() << "]";
                return false;
            }
            edge_dedup.insert(out);
        }
    }

    if (!DiffNodes(node_dedup, all_nodes, topo)) {
        return false;
    }

    set<edgeid_t> all_edges;
    for (auto it = topo->CreateEdgeIter(); it->IsValid(); it->Forward()) {
        all_edges.insert(it->Get()->GetId());
    }
    if (!DiffEdges(edge_dedup, all_edges, topo)) {
        return false;
    }

    return true;
}

}}} // namespace ppl::nn::utils

#endif
