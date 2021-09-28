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

#include "ppl/nn/engines/cuda/optimizer/algos/algo_graph.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/optimizer/algos/algo_filter_manager.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

ir::Node* AlgoGraph::FindBackwardNode(ir::Node* node, uint32_t index) {
    if (node == nullptr) {
        return nullptr;
    }
    auto input_edge = topo_->GetEdgeById(node->GetInput(index));
    return topo_->GetNodeById(input_edge->GetProducer());
}

ir::Node* AlgoGraph::FindForwardNode(ir::Node* node) {
    if (node == nullptr) {
        return nullptr;
    }
    ir::Edge* tensor = topo_->GetEdgeById(node->GetOutput(0));
    auto consumer_iter = tensor->CreateConsumerIter();
    if (!consumer_iter.IsValid()) {
        return nullptr;
    }
    return topo_->GetNodeById(consumer_iter.Get()); // consumer0
}

double AlgoGraph::GetSumTime(AlgoNode* algo_node) {
    double time = 0.0;
    auto time_vect = algo_node->shortest_time;
    for (auto it = time_vect.begin(); it != time_vect.end(); ++it) {
        time += *it;
    }
    return time;
}

uint32_t AlgoGraph::GetConsumerCount(AlgoNode* algo_node, ir::Graph* ir_graph) {
    uint32_t count = 0;
    auto node = algo_node->node;
    for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
        auto edge_id = node->GetOutput(i);
        auto edge = ir_graph->topo->GetEdgeById(edge_id);
        count += edge->CalcConsumerCount();
    }
    return count;
}

RetCode AlgoGraph::CreateNode(ir::Node* node, CudaOptKernel* kernel) {
    AlgoFilterManager* algo_filter_manager = AlgoFilterManager::Instance();
    const AlgoFilter* algo_filter = algo_filter_manager->FindKernel(node->GetType().name);
    if (algo_filter == nullptr) {
        algo_filter = algo_filter_manager->FindKernel("Normal");
    }
    auto formats = algo_filter->GetAlgo(0)->Getformats(node->GetType().name);

    std::vector<AlgoNode*> temp_vect;
    for (auto it = formats.begin(); it != formats.end(); ++it) {
        AlgoNode* algo_node = new AlgoNode();
        // initialize algo_node
        algo_node->node = node;
        algo_node->input_format = it->first;
        algo_node->selected_algo = algo_filter->GetAlgo(0);
        kernel->CopyParam(algo_node->param);

        // initialize shortest time
        for (uint32_t j = 0; j < node->GetInputCount(); ++j) {
            auto edge_id = node->GetInput(j);
            if (edge_id == INVALID_EDGEID) {
                algo_node->shortest_time.emplace_back(0.0f);
            } else {
                algo_node->shortest_time.emplace_back(DOUBLE_MAX);
            }
            algo_node->parents.emplace_back(nullptr);
        }
        temp_vect.emplace_back(algo_node);
    }
    graph_.emplace(node, std::move(temp_vect));
    return RC_SUCCESS;
}

RetCode AlgoGraph::UpdateNode(ir::Node* node, OptKernelOptions& options) {
    AlgoFilterManager* algo_filter_manager = AlgoFilterManager::Instance();
    std::vector<AlgoNode*> temp_vect = graph_.find(node)->second;

    for (uint32_t i = 0; i < node->GetInputCount(); ++i) {
        auto edge_id = node->GetInput(i);
        if (edge_id == INVALID_EDGEID) {
            continue;
        }

        ir::Node* pre_node = FindBackwardNode(node, i);
        if (pre_node == nullptr) { // this is an input node
            auto shape = options.tensors->find(edge_id)->second->GetShape();
            for (auto it = temp_vect.begin(); it != temp_vect.end(); ++it) {
                if (shape.GetDataFormat() == (*it)->input_format) {
                    (*it)->shortest_time[i] = 0.0;
                }
            }
        } else { // this is an internal node
            std::vector<AlgoNode*> pre_vect = graph_.find(pre_node)->second;
            const AlgoFilter* pre_algo_filter = algo_filter_manager->FindKernel(pre_node->GetType().name);
            if (pre_algo_filter == nullptr) {
                pre_algo_filter = algo_filter_manager->FindKernel("Normal");
            }
            auto pre_formats = pre_algo_filter->GetAlgo(0)->Getformats(pre_node->GetType().name);

            bool at_least_one_algo = false;
            for (auto pre_it = pre_vect.begin(); pre_it != pre_vect.end(); ++pre_it) {
                auto sum_time = GetSumTime(*pre_it);
                auto consumer_count = GetConsumerCount(*pre_it, options.graph);

                for (auto it = temp_vect.begin(); it != temp_vect.end(); ++it) {
                    auto output_formats = pre_formats.find((*pre_it)->input_format)->second;
                    if (output_formats.find((*it)->input_format) == output_formats.end()) {
                        continue;
                    }
                    options.param = (*pre_it)->param;
                    for (uint32_t j = 0; j < pre_algo_filter->GetAlgoCount(); ++j) {
                        Algorithm* temp_algo = pre_algo_filter->GetAlgo(j);
                        if (!temp_algo->IsSupported(pre_node, options, (*pre_it)->input_format)) {
                            continue;
                        }
                        at_least_one_algo = true;
                        temp_algo->ReshapeOnEdges(pre_node, options.tensors, (*pre_it)->input_format,
                                                  (*it)->input_format);
                        auto timer = (temp_algo->ExcuteTimer(pre_node, options) + sum_time) / consumer_count;

                        if ((*it)->shortest_time[i] > timer) {
                            (*it)->shortest_time[i] = timer;
                            (*it)->parents[i] = *pre_it;
                            (*pre_it)->selected_algo = temp_algo;
                            temp_algo->GetAttrParam((*pre_it)->param);
                        }
                    }
                }
            }
            if (!at_least_one_algo) {
                LOG(ERROR) << "Can not find any supported algo for node[" << pre_node->GetName() << "].";
                return RC_UNSUPPORTED;
            }
        }
    }

    ir::Node* post_node = FindForwardNode(node);
    if (post_node == nullptr) { // this is an output node
        const AlgoFilter* algo_filter = algo_filter_manager->FindKernel(node->GetType().name);
        if (node->GetType().name != "Bridge") {
            LOG(ERROR) << "The final node is not a bridge node[" << node->GetName() << "]";
            return RC_UNSUPPORTED;
        }

        for (auto it = temp_vect.begin(); it != temp_vect.end(); ++it) {
            auto pre_time = GetSumTime(*it);
            auto postedge_id = node->GetOutput(0);
            auto formats = algo_filter->GetAlgo(0)->Getformats(node->GetType().name);
            auto output_formats = formats.find((*it)->input_format)->second;
            auto output_format = options.tensors->find(postedge_id)->second->GetShape().GetDataFormat();

            if (output_formats.find(output_format) == output_formats.end()) {
                LOG(ERROR) << "Output format " << output_format << " is not match for node[" << node->GetName() << "]";
                return RC_UNSUPPORTED;
            }

            Algorithm* temp_algo = algo_filter->GetAlgo(0);
            for (uint32_t i = 0; i < node->GetInputCount(); ++i)
                (*it)->shortest_time[i] = pre_time;
            (*it)->output_format = output_format;
            (*it)->selected_algo = temp_algo;
        }
    } else if (post_node->GetType().name == "If" ||
               post_node->GetType().name == "Loop") { // sepcial case for if/loop op
        const AlgoFilter* algo_filter = algo_filter_manager->FindKernel("Normal");

        for (auto it = temp_vect.begin(); it != temp_vect.end(); ++it) {
            auto pre_time = GetSumTime(*it);
            Algorithm* temp_algo = algo_filter->GetAlgo(0);
            for (uint32_t i = 0; i < node->GetInputCount(); ++i)
                (*it)->shortest_time[i] = pre_time;
            (*it)->output_format = DATAFORMAT_NDARRAY;
            (*it)->selected_algo = temp_algo;
        }
    }
    return RC_SUCCESS;
}

AlgoNode* AlgoGraph::FindBestAlgo(const ir::Node* node) {
    std::vector<AlgoNode*> temp_vect = graph_.find(node)->second;
    AlgoNode* ans = *temp_vect.begin();
    bool flag = false;
    for (auto it = temp_vect.begin(); it != temp_vect.end(); ++it) {
        if ((*it)->determined) {
            if (!flag) {
                flag = true;
                ans = (*it);
            } else if (GetSumTime(ans) > GetSumTime((*it))) {
                ans = (*it);
            }
        }

        if (!flag && GetSumTime(ans) > GetSumTime((*it))) {
            ans = (*it);
        }
    }
    return ans;
}

RetCode AlgoGraph::DetermineNode(CudaOptKernel* kernel, OptKernelOptions& options) {
    auto node = kernel->GetNode();
    AlgoNode* algo_node = FindBestAlgo(node);
    algo_node->selected_algo->ReshapeOnEdges(node, options.tensors, algo_node->input_format, algo_node->output_format);
    options.param = algo_node->param;
    auto status = algo_node->selected_algo->ModifyParam(node, options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ModifyParam for kernel[" << node->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    status = kernel->Finalize(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Finalize for kernel[" << node->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    for (uint32_t i = 0; i < node->GetInputCount(); ++i) {
        if (algo_node->parents[i] != nullptr) {
            algo_node->parents[i]->determined = true;
            algo_node->parents[i]->output_format = algo_node->input_format;
        }
    }
    return RC_SUCCESS;
}

void AlgoGraph::Delete() {
    for (auto graph_it = graph_.begin(); graph_it != graph_.end(); ++graph_it) {
        for (auto vect_it = graph_it->second.begin(); vect_it != graph_it->second.end(); ++vect_it) {
            if ((*vect_it)->param) {
                (*vect_it)->selected_algo->DeleteAttrParam((*vect_it)->param);
            }
            delete *vect_it;
        }
    }
}

}}} // namespace ppl::nn::cuda
