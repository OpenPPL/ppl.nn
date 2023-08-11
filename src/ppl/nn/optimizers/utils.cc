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

#include "ppl/common/str_utils.h"
#include "ppl/nn/optimizers/utils.h"
#include "ppl/nn/optimizers/generic_optimizer_manager.h"
#include "ppl/nn/engines/common/pmx/converter_op.h"
#include "ppl/nn/engines/utils.h"
#include "ppl/nn/ir/partial_graph_topo.h"
#include "ppl/nn/ir/utils.h"
#include "ppl/nn/runtime/runtime_partition_info.h"
#include "ppl/nn/utils/utils.h"
#include "ppl/nn/common/logger.h"
#include <set>
using namespace std;
using namespace ppl::common;

#ifndef NDEBUG
#include "ppl/nn/auxtools/validate_graph.h"
#endif

namespace ppl { namespace nn { namespace utils {

static nodeid_t AddConverterNode(ir::GraphTopo* topo, const string& name_prefix) {
    const string node_name = name_prefix + "_" + ToString(topo->GetCurrentNodeIdBound());
    auto ret_pair = topo->AddNode(node_name);
    ret_pair.first->SetType(utils::MakePplConverterNodeType());
    return ret_pair.first->GetId();
}

struct ConverterNodeInfo final {
    nodeid_t nid;
    map<edgeid_t, set<nodeid_t>> affected_nodes;
};

static void HandlePartitionInputsWithProducers(uint32_t cur_partidx, const vector<uint32_t>& nid2partidx,
                                               const vector<pair<EngineImpl*, vector<nodeid_t>>>& partitions,
                                               ir::GraphTopo* topo, map<uint32_t, ConverterNodeInfo>* part_cvt_info) {
    auto cur_engine = partitions[cur_partidx].first;

    auto part_inputs = utils::FindInputsOfNodesGroup(topo, partitions[cur_partidx].second);
    for (auto x = part_inputs.begin(); x != part_inputs.end(); ++x) {
        auto eid = *x;
        auto edge = topo->GetEdge(eid);

        auto producer_nid = edge->GetProducer();
        if (producer_nid == INVALID_NODEID) {
            continue;
        }

        auto producer_part_idx = nid2partidx[producer_nid];
        auto producer_engine = partitions[producer_part_idx].first;
        if (producer_engine == cur_engine) {
            continue;
        }

        /*
          If the producer of `edge` is assigned to a different engine from its consumer(s), a converter is required.
          For example, A and B are assigned to one engine and C is assigned to the same engine as the producer node:

                    +----------+                              +----------+
                    | producer |                              | producer |
                    +----------+                              +----------+
                          |                                         |
                          |                                +--------+-------+
                          |                                |                |
                          |               ==>              v                |
                          |                          +===========+          |
                          |                          | converter |          |
                          |                          +===========+          |
                          |                                |                |
                 +--------+--------+                  +----+---+            |
                 |        |        |                  |        |            |
                 v        v        v                  v        v            v
               +===+    +===+    +---+              +===+    +===+        +---+
               | A |    | B |    | C |              | A |    | B |        | C |
               +===+    +===+    +---+              +===+    +===+        +---+

           A and B are affected nodes associated with `edge` in current partition.
        */

        auto ret_pair = part_cvt_info->insert(make_pair(cur_partidx, ConverterNodeInfo()));
        auto& info = ret_pair.first->second;
        if (ret_pair.second) {
            info.nid = AddConverterNode(topo, "__converter_of_partition_" + ToString(cur_partidx));
        }

        auto ret_pair2 = info.affected_nodes.insert(make_pair(eid, set<nodeid_t>()));
        auto& nodes_set = ret_pair2.first->second;
        for (auto it = edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
            auto nid = it.Get();
            if (nid2partidx[nid] == cur_partidx) {
                nodes_set.insert(nid);
            }
        }
    }
}

static void HandleGraphInputs(ir::GraphTopo* topo, const vector<uint32_t>& nid2partidx,
                              const vector<pair<EngineImpl*, vector<nodeid_t>>>& partitions,
                              map<uint32_t, ConverterNodeInfo>* part_cvt_info) {
    set<nodeid_t> begin_nids;
    for (uint32_t i = 0; i < topo->GetInputCount(); ++i) {
        auto eid = topo->GetInput(i);
        auto edge = topo->GetEdge(eid);
        for (auto it = edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
            begin_nids.insert(it.Get());
        }
    }

    map<edgeid_t, EngineImpl*> default_engine_of_graph_input;
    for (auto x = begin_nids.begin(); x != begin_nids.end(); ++x) {
        auto node = topo->GetNode(*x);
        auto part_idx = nid2partidx[*x];
        auto engine = partitions[part_idx].first;

        for (uint32_t i = 0; i < node->GetInputCount(); ++i) {
            auto eid = node->GetInput(i);
            auto edge = topo->GetEdge(eid);
            if (edge->GetProducer() != INVALID_NODEID) {
                continue;
            }

            auto ret_pair = default_engine_of_graph_input.insert(make_pair(eid, engine));
            // different partitions that have the same engine as current input don't need a converter node
            if (ret_pair.second || engine == ret_pair.first->second) {
                continue;
            }

            auto info_ret_pair = part_cvt_info->insert(make_pair(part_idx, ConverterNodeInfo()));
            auto& info = info_ret_pair.first->second;
            if (info_ret_pair.second) {
                info.nid = AddConverterNode(topo, "__converter_of_partition_" + ToString(part_idx));
            }

            auto ret_pair2 = info.affected_nodes.insert(make_pair(eid, set<nodeid_t>()));
            auto& nodes_set = ret_pair2.first->second;
            nodes_set.insert(node->GetId());
        }
    }
}

static RetCode GenConverterNodes(const vector<pair<EngineImpl*, vector<nodeid_t>>>& partitions,
                                 const vector<uint32_t>& nid2partidx, ir::GraphTopo* topo,
                                 map<uint32_t, ConverterNodeInfo>* part_cvt_info) {
    for (uint32_t cur = 0; cur < partitions.size(); ++cur) {
        HandlePartitionInputsWithProducers(cur, nid2partidx, partitions, topo, part_cvt_info);
    }

    HandleGraphInputs(topo, nid2partidx, partitions, part_cvt_info);

    for (auto p = part_cvt_info->begin(); p != part_cvt_info->end(); ++p) {
        auto& info = p->second;
        auto converter_nid = info.nid;
        auto converter_node = topo->GetNode(info.nid);

        for (auto x = info.affected_nodes.begin(); x != info.affected_nodes.end(); ++x) {
            auto eid = x->first;
            auto edge = topo->GetEdge(eid);

            const string output_edge_name("converted_output_of_" + edge->GetName() + "_" +
                                          ToString(topo->GetCurrentEdgeIdBound()));
            auto ret_pair = topo->AddEdge(output_edge_name);
            if (!ret_pair.second) {
                LOG(ERROR) << "add edge[" << output_edge_name << "] failed: exists.";
                return RC_EXISTS;
            }

            auto new_edge = ret_pair.first;
            auto new_eid = new_edge->GetId();

            converter_node->AddInput(eid);
            converter_node->AddOutput(new_eid);

            edge->AddConsumer(converter_nid);
            new_edge->SetProducer(converter_nid);

            for (auto n = x->second.begin(); n != x->second.end(); ++n) {
                edge->DelConsumer(*n);
                new_edge->AddConsumer(*n);

                auto consumer = topo->GetNode(*n);
                consumer->ReplaceInput(eid, new_eid);
                consumer->ReplaceExtraInput(eid, new_eid);
            }
        }
    }

    return RC_SUCCESS;
}

static RetCode CollectOps(const ir::GraphTopo* topo, map<nodeid_t, unique_ptr<OptKernel>>* nid2kernel,
                          vector<unique_ptr<OptKernel>>* ops) {
    vector<unique_ptr<OptKernel>> tmp_ops;
    tmp_ops.reserve(nid2kernel->size());

    for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto nid = it->Get()->GetId();
        auto ref = nid2kernel->find(nid);
        if (ref == nid2kernel->end()) {
            LOG(ERROR) << "cannot find implementation of node[" << it->Get()->GetName() << "]";
            return RC_NOT_FOUND;
        }
        tmp_ops.emplace_back(std::move(ref->second));
    }

    *ops = std::move(tmp_ops);
    return RC_SUCCESS;
}

static RetCode GenPartitionsInfoAndShapes(const vector<pair<EngineImpl*, vector<nodeid_t>>>& partitions,
                                          const utils::SharedResource& resource, ir::Graph* graph,
                                          map<edgeid_t, TensorShape>* shapes,
                                          vector<RuntimeGraphInfo::Partition>* par_list) {
    for (uint32_t p = 0; p < partitions.size(); ++p) {
        auto& partition = partitions[p];
        ir::Graph sub_graph;
        if (partitions.size() == 1) {
            sub_graph.topo = graph->topo;
            sub_graph.data = graph->data;
        } else {
            sub_graph.topo = make_shared<ir::PartialGraphTopo>(graph->topo.get(), partition.second);
            sub_graph.data = graph->data;
            sub_graph.topo->SetName(graph->topo->GetName() + "." + partition.first->GetName() + "." + ToString(p));
        }

        auto engine = partition.first;
        RuntimePartitionInfo subgraph_info;
        auto status = engine->ProcessGraph(resource, &sub_graph, &subgraph_info);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "process graph[" << sub_graph.topo->GetName() << "] by engine[" << engine->GetName()
                       << "] failed: " << GetRetCodeStr(status);
            return status;
        }

        RuntimeGraphInfo::Partition par_info;
        par_info.engine = engine;

        status = CollectOps(sub_graph.topo.get(), &subgraph_info.kernels, &par_info.ops);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "collect optkernels failed: " << GetRetCodeStr(status);
            return status;
        }

        for (auto c = subgraph_info.constants.begin(); c != subgraph_info.constants.end(); ++c) {
            auto ret_pair = par_info.constants.insert(make_pair(c->first, BufferInfo()));
            if (ret_pair.second) {
                auto& src = c->second;
                auto& dst = ret_pair.first->second;
                dst.SetBuffer(src.GetBufferDesc(), src.GetDevice(), src.IsBufferOwner());
                src.DetachBuffer();

                shapes->insert(make_pair(c->first, *src.GetShape()));
            }
        }

        par_list->emplace_back(std::move(par_info));
    }

    return RC_SUCCESS;
}

// TODO optimize
static RetCode CopyConstantsForDevices(const vector<pair<EngineImpl*, vector<nodeid_t>>>& partitions,
                                       const vector<uint32_t>& nid2partidx, ir::Graph* graph) {
    auto topo = graph->topo.get();
    auto graph_data = graph->data.get();

    for (uint32_t i = 0; i < topo->GetConstantCount(); ++i) {
        auto eid = topo->GetConstant(i);
        auto edge = topo->GetEdge(eid);

        if (edge->CalcConsumerCount() == 1) {
            continue;
        }

        map<EngineImpl*, vector<nodeid_t>> engine_node_groups;
        for (auto it = edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
            auto consumer_nid = it.Get();
            auto part_idx = nid2partidx[consumer_nid];
            auto engine = partitions[part_idx].first;
            auto ret_pair = engine_node_groups.insert(make_pair(engine, vector<nodeid_t>()));
            ret_pair.first->second.push_back(consumer_nid);
        }
        if (engine_node_groups.size() <= 1) {
            continue;
        }

        auto constant_data_iter = graph_data->constants.find(eid);
        if (constant_data_iter == graph_data->constants.end()) {
            LOG(ERROR) << "cannot find constant[" << edge->GetName() << "].";
            return RC_NOT_FOUND;
        }

        auto shape_iter = graph_data->shapes.find(eid);
        if (shape_iter == graph_data->shapes.end()) {
            LOG(ERROR) << "cannot find shape of constant[" << edge->GetName() << "].";
            return RC_NOT_FOUND;
        }

        // original constant edge is assigned to one of those engines
        engine_node_groups.erase(engine_node_groups.begin());

        // create copies for other engines
        for (auto it = engine_node_groups.begin(); it != engine_node_groups.end(); ++it) {
            auto ret_pair =
                topo->AddEdge("__copy_of_" + edge->GetName() + "_" + ToString(topo->GetCurrentEdgeIdBound()));
            auto new_edge = ret_pair.first;
            auto new_edge_id = new_edge->GetId();

            topo->MarkAsConstant(new_edge_id);

            graph_data->constants.insert(make_pair(new_edge_id, constant_data_iter->second));
            graph_data->shapes.insert(make_pair(new_edge_id, shape_iter->second));

            // replace inputs and extra inputs of consumers
            for (auto nid = it->second.begin(); nid != it->second.end(); ++nid) {
                auto node = topo->GetNode(*nid);
                new_edge->AddConsumer(*nid);
                edge->DelConsumer(*nid);
                node->ReplaceInput(eid, new_edge_id);
                node->ReplaceExtraInput(eid, new_edge_id);
            }
        }
    }

    return RC_SUCCESS;
}

static vector<uint32_t> GenNid2Partidx(nodeid_t maxid, const vector<pair<EngineImpl*, vector<nodeid_t>>>& partitions) {
    vector<uint32_t> nid2partidx(maxid, UINT32_MAX);
    for (uint32_t i = 0; i < partitions.size(); ++i) {
        auto& vec = partitions[i].second;
        for (uint32_t j = 0; j < vec.size(); ++j) {
            nid2partidx[vec[j]] = i;
        }
    }
    return nid2partidx;
}

RetCode ProcessGraph(const utils::SharedResource& resource, ir::Graph* graph, RuntimeGraphInfo* info) {
    auto status = GenericOptimizerManager::GetInstance()->Process(graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "do optimization failed: " << GetRetCodeStr(status);
        return status;
    }

    vector<pair<EngineImpl*, vector<nodeid_t>>> partitions;
    status = resource.graph_partitioner->Partition(resource.engines, graph->topo.get(), &partitions);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "partitioning graph[" << graph->topo->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    LOG(INFO) << "total partition(s) of graph[" << graph->topo->GetName() << "]: " << partitions.size() << ".";

    map<uint32_t, ConverterNodeInfo> part_cvt_info;
    if (partitions.size() > 1) {
        const vector<uint32_t> nid2partidx = GenNid2Partidx(graph->topo->GetCurrentNodeIdBound(), partitions);

        status = GenConverterNodes(partitions, nid2partidx, graph->topo.get(), &part_cvt_info);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "GenConverterNodes failed: " << GetRetCodeStr(status);
            return status;
        }

        status = CopyConstantsForDevices(partitions, nid2partidx, graph);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "CopyConstantsForDevices failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    /*
      subgraphs MUST be created after inserting converter nodes. because subgraphs cannot visit
      edges that are directly inserted in the main graph.
    */
    status = GenPartitionsInfoAndShapes(partitions, resource, graph, &info->shapes, &info->partitions);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenPartitionsInfoAndShapes failed:" << GetRetCodeStr(status);
        return status;
    }

    // add converter ops to corresponding partitions
    for (auto x = part_cvt_info.begin(); x != part_cvt_info.end(); ++x) {
        info->partitions[x->first].ops.emplace_back(
            unique_ptr<OptKernel>(new pmx::ConverterOp(graph->topo->GetNode(x->second.nid))));
    }

    // save necessary shapes for runtime
    auto topo = graph->topo.get();
    auto graph_data = graph->data.get();
    for (uint32_t i = 0; i < topo->GetInputCount(); ++i) {
        auto ref = graph_data->shapes.find(topo->GetInput(i));
        if (ref != graph_data->shapes.end()) {
            auto ret_pair = info->shapes.insert(make_pair(ref->first, TensorShape()));
            if (ret_pair.second) {
                utils::IrShape2TensorShape(ref->second, &ret_pair.first->second);
            }
        }
    }
    for (uint32_t i = 0; i < topo->GetOutputCount(); ++i) {
        auto ref = graph_data->shapes.find(topo->GetOutput(i));
        if (ref != graph_data->shapes.end()) {
            auto ret_pair = info->shapes.insert(make_pair(ref->first, TensorShape()));
            if (ret_pair.second) {
                utils::IrShape2TensorShape(ref->second, &ret_pair.first->second);
            }
        }
    }

#ifndef NDEBUG
    if (!ValidateGraph(*graph)) {
        LOG(ERROR) << "ValidateGraphTopo failed.";
        return RC_INVALID_VALUE;
    }
#endif

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::utils
