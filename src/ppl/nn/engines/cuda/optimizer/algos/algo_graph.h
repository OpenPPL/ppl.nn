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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_ALGO_GRAPH_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_ALGO_GRAPH_H_

#include <string>

#include "ppl/nn/engines/cuda/optimizer/algos/algorithm.h"

#define DOUBLE_MAX 3.0e+38

namespace ppl { namespace nn { namespace cuda {

struct AlgoNode {
    ir::Node* node;
    Algorithm* selected_algo = nullptr;

    ppl::common::dataformat_t input_format; // record the type of algo_node
    ppl::common::dataformat_t output_format; // only be used for pushing back and determining algorithm
    void* param = nullptr;
    bool determined = false;

    std::vector<double> shortest_time;
    std::vector<AlgoNode*> parents; // depend on the number of inputs of nodes
};

class AlgoGraph {
public:
    AlgoGraph(ir::GraphTopo* topo) : topo_(topo) {}

    ppl::common::RetCode CreateNode(ir::Node* node, CudaOptKernel* kernel);
    ppl::common::RetCode UpdateNode(ir::Node* node, OptKernelOptions& options);
    ppl::common::RetCode DetermineNode(CudaOptKernel* kernel, OptKernelOptions& options);
    AlgoNode* FindBestAlgo(const ir::Node* node);

    void SetInputsShape(ir::Node* node, ir::GraphData* graph_data, ppl::common::dataformat_t format,
                        ppl::common::datatype_t type);
    void SetOutputsShape(ir::Node* node, ir::GraphData* graph_data, ppl::common::dataformat_t format,
                         ppl::common::datatype_t type);
    double GetSumTime(AlgoNode* algo_node);
    uint32_t GetConsumerCount(AlgoNode* algo_node, ir::Graph* ir_graph);
    ir::Node* FindBackwardNode(ir::Node* node, uint32_t index);
    ir::Node* FindForwardNode(ir::Node* node);

    void Delete();

private:
    ir::GraphTopo* topo_;
    std::map<const ir::Node*, std::vector<AlgoNode*>> graph_;
    std::map<const ir::Node*, AlgoNode*> selected_algo_; // store the selected results
};

}}} // namespace ppl::nn::cuda
#endif
