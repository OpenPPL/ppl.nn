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

#include "ppl/nn/optimizers/utils.h"
#include "ppl/nn/optimizers/engine_graph_partitioner.h"
#include "ppl/nn/optimizers/graph_optimizer_manager.h"
#include "ppl/nn/engines/common/ppl/converter_op.h"
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

static vector<EngineImpl*> GenNode2Engine(const vector<pair<EngineImpl*, vector<nodeid_t>>>& partitions,
                                          uint32_t max_node_id) {
    vector<EngineImpl*> node2engine(max_node_id, nullptr);

    for (auto it = partitions.begin(); it != partitions.end(); ++it) {
        for (auto x = it->second.begin(); x != it->second.end(); ++x) {
            node2engine[*x] = it->first;
        }
    }

    return node2engine;
}

static ir::Edge* CreateEdge(const string& name_prefix, ir::GraphTopo* topo) {
    uint32_t suffix_num = 0;
    pair<ir::Edge*, bool> ret_pair = {nullptr, false};
    do {
        // older android ndk does not support std::to_string
        char buf[8];
        int buf_len = sprintf(buf, "%u", suffix_num);
        const string edge_name = name_prefix + "_" + string(buf, buf_len);
        ++suffix_num;
        if (suffix_num > 1000) {
            LOG(ERROR) << "create edge[" << name_prefix << "] failed after trying 1000 times.";
            return nullptr;
        }

        ret_pair = topo->AddEdge(edge_name);
    } while (!ret_pair.second);

    return ret_pair.first;
}

static ir::Node* CreateConverterNode(const string& name_prefix, ir::GraphTopo* topo) {
    pair<ir::Node*, bool> ret_pair = {nullptr, false};

    uint32_t suffix_num = 0;
    do {
        // older android ndk does not support std::to_string
        char buf[8];
        int buf_len = sprintf(buf, "%u", suffix_num);
        const string node_name(name_prefix + "_" + string(buf, buf_len));
        ret_pair = topo->AddNode(node_name);
        ++suffix_num;
        if (suffix_num > 1000) {
            LOG(ERROR) << "create node failed after trying 1000 times.";
            return nullptr;
        }
    } while (!ret_pair.second);

    auto converter_node = ret_pair.first;
    converter_node->SetType(utils::MakePplConverterNodeType());
    return converter_node;
}

static RetCode CreateNewOutputs(const vector<edgeid_t>& edges, ir::GraphTopo* topo, vector<edgeid_t>* new_outputs) {
    for (auto x = edges.begin(); x != edges.end(); ++x) {
        auto edge = topo->GetEdgeById(*x);
        auto new_edge = CreateEdge("converted_output_for_" + edge->GetName(), topo);
        if (!new_edge) {
            LOG(ERROR) << "create converted input for edge[" << edge->GetName() << "] failed.";
            return RC_OTHER_ERROR;
        }

        new_outputs->push_back(new_edge->GetId());
    }

    return RC_SUCCESS;
}

// an edge may appear in different input slots of a node
static void CollectCommonEdges(const set<edgeid_t>& parent_outputs, const vector<nodeid_t>& successors,
                               const ir::GraphTopo* topo, vector<edgeid_t>* common_edges) {
    common_edges->reserve(parent_outputs.size());
    for (auto x = successors.begin(); x != successors.end(); ++x) {
        auto successor = topo->GetNodeById(*x);
        for (uint32_t i = 0; i < successor->GetInputCount(); ++i) {
            auto eid = successor->GetInput(i);
            if (parent_outputs.find(eid) != parent_outputs.end()) {
                utils::VectorAddUnique(*common_edges, eid);
            }
        }
        for (uint32_t i = 0; i < successor->GetExtraInputCount(); ++i) {
            auto eid = successor->GetExtraInput(i);
            if (parent_outputs.find(eid) != parent_outputs.end()) {
                utils::VectorAddUnique(*common_edges, eid);
            }
        }
    }
}

static ir::Node* AddConverterNodeAndOutputs(const vector<edgeid_t>& common_edges, const vector<nodeid_t>& successors,
                                            const string& name_prefix, ir::GraphTopo* topo) {
    auto converter_node = CreateConverterNode(name_prefix, topo);
    if (!converter_node) {
        LOG(ERROR) << "create converter node [" << name_prefix << "] failed.";
        return nullptr;
    }

    vector<edgeid_t> new_outputs;
    auto status = CreateNewOutputs(common_edges, topo, &new_outputs);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "CreateNewOutputs of [" << name_prefix << "] failed.";
        return nullptr;
    }

    // set converter node's inputs and outputs
    for (uint32_t i = 0; i < common_edges.size(); ++i) {
        converter_node->AddInput(common_edges[i]);
        converter_node->AddOutput(new_outputs[i]);

        auto in_edge = topo->GetEdgeById(common_edges[i]);
        in_edge->AddConsumer(converter_node->GetId());

        auto out_edge = topo->GetEdgeById(new_outputs[i]);
        out_edge->SetProducer(converter_node->GetId());
    }

    for (auto x = successors.begin(); x != successors.end(); ++x) {
        auto successor = topo->GetNodeById(*x);

        for (uint32_t i = 0; i < common_edges.size(); ++i) {
            uint32_t count_a = successor->ReplaceInput(common_edges[i], new_outputs[i]);
            uint32_t count_b = successor->ReplaceExtraInput(common_edges[i], new_outputs[i]);
            if (count_a > 0 || count_b > 0) {
                auto in_edge = topo->GetEdgeById(common_edges[i]);
                in_edge->DelConsumer(*x);
                auto out_edge = topo->GetEdgeById(new_outputs[i]);
                out_edge->AddConsumer(*x);
            }
        }
    }

    return converter_node;
}

static RetCode InsertConverterNodesForPartitions(const vector<EngineImpl*>& node2engine, ir::Graph* graph,
                                                 vector<pair<nodeid_t, EngineImpl*>>* converter_nodes) {
    auto topo = graph->topo.get();
    const nodeid_t max_node_id = topo->GetMaxNodeId();

    // newly inserted converter nodes' ids start from max_node_id
    for (nodeid_t nid = 0; nid < max_node_id; ++nid) {
        auto parent = topo->GetNodeById(nid);
        if (!parent) {
            continue;
        }

        auto successor_ids = topo->FindSuccessors(nid);
        if (successor_ids.empty()) {
            continue;
        }

        auto producer_engine = node2engine[nid];
        map<EngineImpl*, vector<nodeid_t>> engine_node_groups;
        for (auto x = successor_ids.begin(); x != successor_ids.end(); ++x) {
            auto engine = node2engine[*x];
            if (engine != producer_engine) {
                auto ret_pair = engine_node_groups.insert(make_pair(engine, vector<nodeid_t>()));
                ret_pair.first->second.push_back(*x);
            }
        }
        if (engine_node_groups.empty()) {
            continue;
        }

        set<edgeid_t> parent_outputs;
        for (uint32_t j = 0; j < parent->GetOutputCount(); ++j) {
            parent_outputs.insert(parent->GetOutput(j));
        }

        for (auto x = engine_node_groups.begin(); x != engine_node_groups.end(); ++x) {
            // common edges between parent and successors that are assigned to this engine
            vector<edgeid_t> common_edges;
            CollectCommonEdges(parent_outputs, x->second, topo, &common_edges);

            auto converter_node =
                AddConverterNodeAndOutputs(common_edges, x->second, "converter_of_" + parent->GetName(), topo);
            if (!converter_node) {
                LOG(ERROR) << "create converter node for [" << parent->GetName() << "] failed.";
                return RC_OTHER_ERROR;
            }

            converter_nodes->push_back(make_pair(converter_node->GetId(), x->first));
        }
    }

    return RC_SUCCESS;
}

static RetCode CollectSortedOps(const ir::GraphTopo* topo, RuntimePartitionInfo* sub_info,
                                vector<unique_ptr<OptKernel>>* ops) {
    vector<nodeid_t> sub_sorted_nodes;
    topo->TopologicalSort([&sub_sorted_nodes](nodeid_t nid) -> void {
        sub_sorted_nodes.push_back(nid);
    });

    vector<unique_ptr<OptKernel>> tmp_ops(sub_sorted_nodes.size());
    for (uint32_t i = 0; i < sub_sorted_nodes.size(); ++i) {
        auto ref = sub_info->kernels.find(sub_sorted_nodes[i]);
        if (ref == sub_info->kernels.end()) {
            auto node = topo->GetNodeById(sub_sorted_nodes[i]);
            LOG(ERROR) << "cannot find node[" << node->GetName() << "] in opt kernels";
            return RC_NOT_FOUND;
        }
        tmp_ops[i] = std::move(ref->second);
        sub_info->kernels.erase(ref);
    }

    *ops = std::move(tmp_ops);
    return RC_SUCCESS;
}

static RetCode GenPartitionsInfo(const vector<pair<EngineImpl*, vector<nodeid_t>>>& partitions,
                                 utils::SharedResource* resource, ir::Graph* graph,
                                 vector<pair<edgeid_t, RuntimeConstantInfo>>* constants,
                                 vector<RuntimeGraphInfo::Partition>* info_list) {
    for (uint32_t p = 0; p < partitions.size(); ++p) {
        auto& partition = partitions[p];
        ir::Graph sub_graph;
        if (partitions.size() == 1) {
            sub_graph.topo = graph->topo;
            sub_graph.data = graph->data;
        } else {
            sub_graph.topo = make_shared<ir::PartialGraphTopo>(
                graph->topo.get(), graph->topo->GetName() + "." + partition.first->GetName() + "." + std::to_string(p),
                partition.second);
            sub_graph.data = graph->data;
        }

        auto engine = partition.first;
        RuntimePartitionInfo subgraph_info;
        auto status = engine->ProcessGraph(resource, &sub_graph, &subgraph_info);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "process graph[" << sub_graph.topo->GetName() << "] by engine[" << engine->GetName()
                       << "] failed: " << GetRetCodeStr(status);
            return status;
        }

        for (auto x = subgraph_info.constants.begin(); x != subgraph_info.constants.end(); ++x) {
            constants->emplace_back(x->first, std::move(x->second));
        }

        RuntimeGraphInfo::Partition par_info;
        par_info.engine = engine;
        status = CollectSortedOps(sub_graph.topo.get(), &subgraph_info, &par_info.sorted_ops);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "collect optkernels failed: " << GetRetCodeStr(status);
            return status;
        }
        info_list->emplace_back(std::move(par_info));
    }

    return RC_SUCCESS;
}

// TODO optimize
static RetCode CopyConstantsForDevices(const vector<EngineImpl*>& node2engine, ir::Graph* graph) {
    auto topo = graph->topo.get();
    auto graph_data = graph->data.get();

    for (uint32_t i = 0; i < topo->GetConstantCount(); ++i) {
        auto eid = topo->GetConstant(i);
        auto edge = topo->GetEdgeById(eid);

        if (edge->CalcConsumerCount() == 1) {
            continue;
        }

        map<EngineImpl*, vector<nodeid_t>> engine_node_groups;
        for (auto it = edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
            auto consumer_id = it.Get();
            auto engine = node2engine[consumer_id];
            auto ret_pair = engine_node_groups.insert(make_pair(engine, vector<nodeid_t>()));
            ret_pair.first->second.push_back(consumer_id);
        }
        if (engine_node_groups.size() == 1) {
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
            auto new_edge = CreateEdge(string(it->first->GetName()) + "_" + edge->GetName(), topo);
            if (!new_edge) {
                LOG(ERROR) << "CreateEdge failed";
                return RC_OTHER_ERROR;
            }
            auto new_edge_id = new_edge->GetId();

            graph_data->constants.insert(make_pair(new_edge_id, constant_data_iter->second));
            graph_data->shapes.insert(make_pair(new_edge_id, shape_iter->second));

            // replace inputs and extra inputs of consumers
            for (auto nid = it->second.begin(); nid != it->second.end(); ++nid) {
                auto node = topo->GetNodeById(*nid);
                new_edge->AddConsumer(*nid);
                edge->DelConsumer(*nid);
                node->ReplaceInput(eid, new_edge_id);
                node->ReplaceExtraInput(eid, new_edge_id);
            }
        }
    }

    return RC_SUCCESS;
}

static RetCode InsertConverterNodesForInputs(ir::Graph* graph, const vector<EngineImpl*>& node2engine,
                                             vector<pair<nodeid_t, EngineImpl*>>* converter_nodes) {
    auto topo = graph->topo.get();
    map<EngineImpl*, nodeid_t> engine_converter_nodes;

    for (uint32_t i = 0; i < topo->GetInputCount(); ++i) {
        auto eid = topo->GetInput(i);
        auto edge = topo->GetEdgeById(eid);

        if (edge->CalcConsumerCount() <= 1) {
            continue;
        }

        map<EngineImpl*, vector<nodeid_t>> engine_node_groups;
        for (auto it = edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
            auto consumer_id = it.Get();
            auto engine = node2engine[consumer_id];
            auto ret_pair = engine_node_groups.insert(make_pair(engine, vector<nodeid_t>()));
            ret_pair.first->second.push_back(consumer_id);
        }
        if (engine_node_groups.size() == 1) {
            continue;
        }

        /*
          choose one of those engines as the producer engine of this input.
          this will be done when runtime is created.
        */
        engine_node_groups.erase(engine_node_groups.begin());

        for (auto it = engine_node_groups.begin(); it != engine_node_groups.end(); ++it) {
            auto engine = it->first;
            ir::Node* converter_node = nullptr;

            auto ret_pair = engine_converter_nodes.insert(make_pair(it->first, INVALID_NODEID));
            if (ret_pair.second) {
                const string name_prefix = "converter_of_" + string(engine->GetName());
                converter_node = CreateConverterNode(name_prefix, topo);
                if (!converter_node) {
                    LOG(ERROR) << "create converter node [" << name_prefix << "] failed.";
                    return RC_OTHER_ERROR;
                }
                ret_pair.first->second = converter_node->GetId();
                converter_nodes->push_back(make_pair(converter_node->GetId(), engine));
            } else {
                converter_node = topo->GetNodeById(ret_pair.first->second);
            }

            auto new_output = CreateEdge("converted_output_of_" + edge->GetName(), topo);
            if (!new_output) {
                LOG(ERROR) << "create converted input for edge[" << edge->GetName() << "] failed.";
                return RC_OTHER_ERROR;
            }
            auto new_output_id = new_output->GetId();

            converter_node->AddInput(eid);
            converter_node->AddOutput(new_output_id);

            edge->AddConsumer(converter_node->GetId());
            new_output->SetProducer(converter_node->GetId());

            for (auto x = it->second.begin(); x != it->second.end(); ++x) {
                edge->DelConsumer(*x);
                new_output->AddConsumer(*x);

                auto consumer = topo->GetNodeById(*x);
                consumer->ReplaceInput(eid, new_output_id);
                consumer->ReplaceExtraInput(eid, new_output_id);
            }
        }
    }

    return RC_SUCCESS;
}

RetCode ProcessGraph(utils::SharedResource* resource, ir::Graph* graph, RuntimeGraphInfo* info) {
    GraphOptimizerManager optimizer_mgr;
    auto status = optimizer_mgr.Process(graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "do optimization failed: " << GetRetCodeStr(status);
        return status;
    }

    vector<pair<EngineImpl*, vector<nodeid_t>>> partitions;

    EngineGraphPartitioner partitioner;
    status = partitioner.Partition(resource, graph->topo.get(), &partitions);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "partitioning graph[" << graph->topo->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    vector<pair<nodeid_t, EngineImpl*>> converter_nodes;
    if (partitions.size() > 1) {
        auto node2engine = GenNode2Engine(partitions, graph->topo->GetMaxNodeId());

        status = InsertConverterNodesForPartitions(node2engine, graph, &converter_nodes);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "insert converter node failed: " << GetRetCodeStr(status);
            return status;
        }

        status = CopyConstantsForDevices(node2engine, graph);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "CopyConstantsForDevices failed: " << GetRetCodeStr(status);
            return status;
        }

        status = InsertConverterNodesForInputs(graph, node2engine, &converter_nodes);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "InsertConverterNodesForInputs failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    /*
      subgraphs MUST be created after inserting converter nodes. because subgraphs cannot visit
      edges that are directly inserted in the main graph.
    */
    status = GenPartitionsInfo(partitions, resource, graph, &info->constants, &info->partitions);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenPartitionsInfo failed:" << GetRetCodeStr(status);
        return status;
    }

    // add converter optkernels
    for (auto x = converter_nodes.begin(); x != converter_nodes.end(); ++x) {
        RuntimeGraphInfo::Partition par_info;
        par_info.engine = x->second;
        par_info.sorted_ops.emplace_back(
            unique_ptr<OptKernel>(new common::ConverterOp(graph->topo->GetNodeById(x->first))));
        info->partitions.emplace_back(std::move(par_info)); // one converter is treated as a single partition
    }

    // save input shapes for runtime
    auto graph_data = graph->data.get();
    for (auto it = graph_data->shapes.begin(); it != graph_data->shapes.end(); ++it) {
        TensorShape tensor_shape;
        utils::IrShape2TensorShape(it->second, &tensor_shape);
        info->shapes.insert(make_pair(it->first, tensor_shape));
    }

#ifndef NDEBUG
    if (!ValidateGraphTopo(graph->topo.get())) {
        LOG(ERROR) << "ValidateGraphTopo failed.";
        return RC_INVALID_VALUE;
    }
#endif

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::utils
