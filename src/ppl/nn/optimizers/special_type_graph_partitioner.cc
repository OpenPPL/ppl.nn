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

#include "ppl/nn/optimizers/special_type_graph_partitioner.h"
#include "ppl/nn/ir/partial_graph_topo.h"
#include "ppl/nn/ir/utils.h"
#include "ppl/nn/common/logger.h"
#include <set>
#include <queue>
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

static EngineImpl* FindEngine(const vector<EngineImpl*>& engines, const ir::Node* node) {
    for (auto it = engines.begin(); it != engines.end(); ++it) {
        auto engine = *it;
        if (engine->Supports(node)) {
            return engine;
        }
    }

    return nullptr;
}

struct PartitionInfo final {
    // whether nodes are special_types_
    bool is_special_types = false;
    EngineImpl* engine;
    vector<nodeid_t> nodes;
};

// returns the partition index if all predecessors are in the same partition, otherwise returns UINT32_MAX
static uint32_t PredecessorsInTheSamePartition(const vector<nodeid_t>& prev_ids,
                                               const map<nodeid_t, uint32_t>& nid2par) {
    uint32_t par_idx = UINT32_MAX;
    for (auto x = prev_ids.begin(); x != prev_ids.end(); ++x) {
        auto ref = nid2par.find(*x);
        if (par_idx == UINT32_MAX) {
            par_idx = ref->second;
        } else if (ref->second != par_idx) {
            return UINT32_MAX;
        }
    }
    return par_idx;
}

static vector<nodeid_t> FindLowestCommonAncestors(nodeid_t a, nodeid_t b, const ir::GraphTopo* topo) {
    set<nodeid_t> ancestors_of_a = topo->FindAncestors(a);
    set<nodeid_t> ancestors_of_b = topo->FindAncestors(b);
    vector<nodeid_t> common_nodes(ancestors_of_a.size() + ancestors_of_b.size());
    auto end_iter = std::set_intersection(ancestors_of_a.begin(), ancestors_of_a.end(), ancestors_of_b.begin(),
                                          ancestors_of_b.end(), common_nodes.begin());
    common_nodes.resize(end_iter - common_nodes.begin());
    ir::PartialGraphTopo sub_topo(const_cast<ir::GraphTopo*>(topo), "", common_nodes);
    vector<nodeid_t> ret;
    for (uint32_t i = 0; i < sub_topo.GetOutputCount(); ++i) {
        ret.push_back(sub_topo.GetOutput(i));
    }
    return ret;
}

static bool IsInTheSamePartition(uint32_t par_idx, const vector<nodeid_t>& common_ancestors,
                                 const map<nodeid_t, uint32_t>& nid2par) {
    for (auto x = common_ancestors.begin(); x != common_ancestors.end(); ++x) {
        auto par_prev_idx_ref = nid2par.find(*x);
        if (par_prev_idx_ref->second == par_idx) {
            return true;
        }
    }
    return false;
}
// returns prev id, or INVALID_NODEID if none.
static nodeid_t WhichPrevToJoin(const ir::GraphTopo* topo, const EngineImpl* engine, const vector<nodeid_t>& prev_ids,
                                const vector<PartitionInfo>& par_infos, const map<nodeid_t, uint32_t>& nid2par) {
    vector<bool> can_be_merged_with(prev_ids.size(), true);

    for (uint32_t i = 0; i < prev_ids.size(); ++i) {
        if (!can_be_merged_with[i]) {
            continue;
        }

        uint32_t prev_par_idx = nid2par.find(prev_ids[i])->second;
        auto& prev_par_info = par_infos[prev_par_idx];
        if (engine != prev_par_info.engine) {
            can_be_merged_with[i] = false;
            continue;
        }

        for (uint32_t j = i + 1; j < prev_ids.size(); ++j) {
            /*
              if `prev-1` is in the same partition as one of the common ancestors,
              merging current node into that partition will form circular dependencies with
              the partition which contains `prev-2`.

               +--------------------------------------------------+
               | common ancestors of predecessors of current node |
               +--------------------------------------------------+
                                        |
                         +--------------+------------+
                         |              |            |
                         |           +++++++         |
                         |           | ... |         |
                         |           +++++++         |
                         |              |            |
                     +--------+     +========+     +-----+
                     | prev-1 |     | prev-2 |     | ... |
                     +--------+     +========+     +-----+
                        |               |            |
                        +---------------+------------+
                                        |
                                +--------------+
                                | current node |
                                +--------------+
            */
            auto common_ancestors = FindLowestCommonAncestors(prev_ids[i], prev_ids[j], topo);
            if (IsInTheSamePartition(prev_par_idx, common_ancestors, nid2par)) {
                can_be_merged_with[i] = false;
                break;
            }

            auto prev2_par_idx = nid2par.find(prev_ids[j])->second;
            if (IsInTheSamePartition(prev2_par_idx, common_ancestors, nid2par)) {
                can_be_merged_with[j] = false;
                break;
            }
        }
    }

    for (uint32_t i = 0; i < can_be_merged_with.size(); ++i) {
        if (can_be_merged_with[i]) {
            return prev_ids[i];
        }
    }
    return INVALID_NODEID;
}

// TODO optimize: use an alternative engine of ops in FindEngine()
RetCode SpecialTypeGraphPartitioner::Partition(const vector<EngineImpl*>& engines, const ir::GraphTopo* topo,
                                               vector<pair<EngineImpl*, vector<nodeid_t>>>* partitions) const {
    vector<PartitionInfo> par_infos;
    map<nodeid_t, uint32_t> nid2par; // nodeid => index of par_infos

    vector<nodeid_t> sorted_nodes;
    utils::DfsDeeperFirst(topo, [&sorted_nodes](nodeid_t nid) -> void {
        sorted_nodes.push_back(nid);
    });

    vector<uint32_t> root_partitions_idx;

    for (uint32_t i = 0; i < sorted_nodes.size(); ++i) {
        auto nid = sorted_nodes[i];
        auto node = topo->GetNodeById(nid);

        auto engine = FindEngine(engines, node);
        if (!engine) {
            auto& type = node->GetType();
            LOG(ERROR) << "op[" << node->GetName() << "] with type[" << type.domain << ":" << type.name << ":"
                       << type.version << "] is not supported.";
            return RC_UNSUPPORTED;
        }

        bool is_special_type = IsSpecialType(node->GetType());
        auto prev_ids = topo->FindPredecessors(nid);
        if (prev_ids.empty()) {
            bool is_inserted = false;
            // try to merge a root node to existing root partitions
            for (auto p = root_partitions_idx.begin(); p != root_partitions_idx.end(); ++p) {
                auto& partition = par_infos[*p];
                if (is_special_type == partition.is_special_types && engine == partition.engine) {
                    partition.nodes.push_back(nid);
                    nid2par.insert(make_pair(nid, *p));
                    is_inserted = true;
                    break;
                }
            }
            if (!is_inserted) {
                PartitionInfo new_par;
                new_par.is_special_types = is_special_type;
                new_par.engine = engine;
                new_par.nodes.push_back(nid);
                auto new_par_idx = par_infos.size();
                par_infos.emplace_back(std::move(new_par));
                nid2par.insert(make_pair(nid, new_par_idx));
                root_partitions_idx.push_back(new_par_idx);
            }
        } else {
            auto prev_par_idx = PredecessorsInTheSamePartition(prev_ids, nid2par);

            // a node can be added to a partition that all of its predecessors belong to
            if (prev_par_idx != UINT32_MAX) {
                auto& prev_par = par_infos[prev_par_idx];
                if (is_special_type == prev_par.is_special_types && engine == prev_par.engine) {
                    prev_par.nodes.push_back(nid);
                    nid2par.insert(make_pair(nid, prev_par_idx));
                } else {
                    // creates a new partition for different types of consecutive nodes
                    PartitionInfo new_par;
                    new_par.is_special_types = is_special_type;
                    new_par.engine = engine;
                    new_par.nodes.push_back(nid);
                    par_infos.emplace_back(std::move(new_par));
                    nid2par.insert(make_pair(nid, par_infos.size() - 1));
                }
            } else {
                auto prev_id = WhichPrevToJoin(topo, engine, prev_ids, par_infos, nid2par);
                if (prev_id == INVALID_NODEID) {
                    PartitionInfo new_par;
                    new_par.is_special_types = is_special_type;
                    new_par.engine = engine;
                    new_par.nodes.push_back(nid);
                    par_infos.emplace_back(std::move(new_par));
                    nid2par.insert(make_pair(nid, par_infos.size() - 1));
                } else {
                    auto ref = nid2par.find(prev_id);
                    auto& prev_par = par_infos[ref->second];
                    if (is_special_type == prev_par.is_special_types && engine == prev_par.engine) {
                        prev_par.nodes.push_back(nid);
                        nid2par.insert(make_pair(nid, ref->second));
                    } else {
                        PartitionInfo new_par;
                        new_par.is_special_types = is_special_type;
                        new_par.engine = engine;
                        new_par.nodes.push_back(nid);
                        par_infos.emplace_back(std::move(new_par));
                        nid2par.insert(make_pair(nid, par_infos.size() - 1));
                    }
                }
            }
        }
    }

    partitions->reserve(par_infos.size());
    for (auto p = par_infos.begin(); p != par_infos.end(); ++p) {
        partitions->emplace_back(p->engine, std::move(p->nodes));
    }

    return RC_SUCCESS;
}

}} // namespace ppl::nn
