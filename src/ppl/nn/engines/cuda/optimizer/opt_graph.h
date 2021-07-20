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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPT_GRAPH_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPT_GRAPH_H_

#include <map>
#include <set>
#include <memory>

#include "ppl/nn/engines/cuda/optimizer/fusions/fs_filter_manager.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/engines/cuda/engine.h"
#include "ppl/nn/ir/graph.h"
#include "ppl/nn/runtime/runtime_graph_info.h"
#include "ppl/nn/utils/generic_cpu_device.h"

namespace ppl { namespace nn { namespace cuda {

class OptGraph final {
public:
    OptGraph(ir::Graph* graph, RuntimePartitionInfo* info, utils::SharedResource* resource, CudaArgs* args)
        : graph_(graph), info_(info), resource_(resource), args_(args) {}
    ~OptGraph();

    ppl::common::RetCode DoOptimize(CudaDevice*);

private:
    ppl::common::RetCode InitKernels();
    ppl::common::RetCode UpdateDims();
    ppl::common::RetCode FuseOperator();
    ppl::common::RetCode AddBridgeKernels();
    ppl::common::RetCode UpdateType();
    ppl::common::RetCode SelectAlgos(CudaDevice*);
    ppl::common::RetCode LoadConstants(CudaDevice*);
    ppl::common::RetCode DeleteBridgeKernels();
    int32_t LastLegalNodeIndex();
    void UpdateTopologicalSort();

private:
    ir::Graph* graph_;
    RuntimePartitionInfo* info_;
    utils::SharedResource* resource_;
    CudaArgs* args_;
    std::set<nodeid_t> illegal_dims_;
    utils::GenericCpuDevice default_cpu_device_;
    std::map<edgeid_t, std::unique_ptr<TensorImpl>> tensor_impls_;
    std::vector<nodeid_t> sorted_node_ids_;
};

}}} // namespace ppl::nn::cuda

#endif
