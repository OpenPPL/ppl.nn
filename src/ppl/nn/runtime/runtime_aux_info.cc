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
#include "ppl/nn/ir/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

/** @brief calculate each edge's reference count in `topo`. */
static vector<uint32_t> CalcEdgeRefcount(const ir::GraphTopo* topo) {
    vector<uint32_t> edge_refcount(topo->GetMaxEdgeId(), 0);

    for (auto it = topo->CreateEdgeIter(); it->IsValid(); it->Forward()) {
        auto edge = it->Get();

        uint32_t refcount = 0;
        for (auto iter = edge->CreateConsumerIter(); iter.IsValid(); iter.Forward()) {
            auto next = topo->GetNodeById(iter.Get());
            for (uint32_t i = 0; i < next->GetInputCount(); ++i) {
                auto eid = next->GetInput(i);
                if (eid == edge->GetId()) {
                    ++refcount;
                }
            }
            for (uint32_t i = 0; i < next->GetExtraInputCount(); ++i) {
                auto eid = next->GetExtraInput(i);
                if (eid == edge->GetId()) {
                    ++refcount;
                }
            }
        }

        /*
          edge_refcount = consumer_count + producer_count
          if a object's producer does not exist, which means that it is an input object,
          we increase its refcount to make sure that the refcount will always > 0 during
          runtime.
        */
        edge_refcount[edge->GetId()] = refcount + 1 /* for producer */;
    }

    /*
      inputs/extra_inputs/outputs/constants cannot be freed during Run(),
      we increase their refcounts to ensure that their refcounts will always > 0
      during runtime.
    */
    for (uint32_t i = 0; i < topo->GetInputCount(); ++i) {
        auto eid = topo->GetInput(i);
        ++edge_refcount[eid];
    }
    for (uint32_t i = 0; i < topo->GetExtraInputCount(); ++i) {
        auto eid = topo->GetExtraInput(i);
        ++edge_refcount[eid];
    }
    for (uint32_t i = 0; i < topo->GetConstantCount(); ++i) {
        auto eid = topo->GetConstant(i);
        ++edge_refcount[eid];
    }
    for (uint32_t i = 0; i < topo->GetOutputCount(); ++i) {
        auto eid = topo->GetOutput(i);
        ++edge_refcount[eid];
    }

    return edge_refcount;
}

static RetCode InitTensorLastConsumer(const ir::GraphTopo* topo, const vector<nodeid_t>& snodes,
                                      vector<nodeid_t>* tensor_last_consumer) {
    tensor_last_consumer->resize(topo->GetMaxEdgeId(), topo->GetMaxNodeId());

    auto sorted_nodes = snodes;
    auto edge_refcount = CalcEdgeRefcount(topo);

    for (auto x = sorted_nodes.begin(); x != sorted_nodes.end(); ++x) {
        auto node = topo->GetNodeById(*x);

        for (uint32_t i = 0; i < node->GetInputCount(); ++i) {
            auto eid = node->GetInput(i);
            if (eid >= edge_refcount.size()) {
                continue;
            }

            uint32_t& rc = edge_refcount[eid];
            if (rc > 0) {
                --rc;
                if (rc == 0) {
                    tensor_last_consumer->at(eid) = node->GetId();
                }
            } else {
                LOG(ERROR) << "invalid refcount of edge[" << topo->GetEdgeById(eid)->GetName() << "]";
                return RC_INVALID_VALUE;
            }
        }

        for (uint32_t i = 0; i < node->GetExtraInputCount(); ++i) {
            auto eid = node->GetExtraInput(i);
            if (eid >= edge_refcount.size()) {
                continue;
            }

            uint32_t& rc = edge_refcount[eid];
            if (rc > 0) {
                --rc;
                if (rc == 0) {
                    tensor_last_consumer->at(eid) = node->GetId();
                }
            } else {
                LOG(ERROR) << "invalid refcount of edge[" << topo->GetEdgeById(eid)->GetName() << "]";
                return RC_INVALID_VALUE;
            }
        }

        for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
            auto eid = node->GetOutput(i);
            uint32_t& rc = edge_refcount[eid];
            if (rc > 0) {
                --rc;
                if (rc == 0) {
                    tensor_last_consumer->at(eid) = node->GetId();
                }
            } else {
                LOG(ERROR) << "invalid refcount of edge[" << topo->GetEdgeById(eid)->GetName() << "]";
                return RC_INVALID_VALUE;
            }
        }
    }

    return RC_SUCCESS;
}

static vector<nodeid_t> SortNodes(const ir::GraphTopo* topo) {
    // get node levels for second stage
    vector<uint32_t> nid2level(topo->GetMaxNodeId(), 0);
    auto node_iter = topo->CreateNodeIter();
    utils::Bfs(
        topo->GetMaxNodeId(),
        [&node_iter]() -> nodeid_t {
            if (node_iter->IsValid()) {
                auto node = node_iter->Get();
                node_iter->Forward();
                return node->GetId();
            }
            return INVALID_NODEID;
        },
        [topo](nodeid_t nid, const function<void(nodeid_t)>& f) -> void {
            auto prevs = topo->FindPredecessors(nid);
            for (auto x : prevs) {
                f(x);
            }
        },
        [topo](nodeid_t nid, const function<void(nodeid_t)>& f) -> void {
            auto nexts = topo->FindSuccessors(nid);
            for (auto x : nexts) {
                f(x);
            }
        },
        [topo, &nid2level](nodeid_t nid, uint32_t level) -> void {
            nid2level[nid] = level;
        });

    vector<nodeid_t> ret;
    node_iter = topo->CreateNodeIter();
    utils::Dfs(
        topo->GetMaxNodeId(),
        [&node_iter, &topo]() -> nodeid_t {
            if (node_iter->IsValid()) {
                auto node = node_iter->Get();
                node_iter->Forward();
                return node->GetId();
            }
            return INVALID_NODEID;
        },
        [topo](nodeid_t nid, const function<void(nodeid_t)>& f) -> void {
            auto prevs = topo->FindPredecessors(nid);
            for (auto x : prevs) {
                f(x);
            }
        },
        [&ret](nodeid_t nid) -> void {
            ret.push_back(nid);
        },
        [&nid2level](nodeid_t a, nodeid_t b) -> bool {
            // nodes in the longer path will be evaluated first
            return (nid2level[a] < nid2level[b]);
        });

    return ret;
}

RetCode GenerateRuntimeAuxInfo(const ir::GraphTopo* topo, RuntimeAuxInfo* info) {
    info->sorted_nodes = SortNodes(topo);

    auto status = InitTensorLastConsumer(topo, info->sorted_nodes, &info->tensor_last_consumer);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "InitTensorLastConsumer failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

}} // namespace ppl::nn
