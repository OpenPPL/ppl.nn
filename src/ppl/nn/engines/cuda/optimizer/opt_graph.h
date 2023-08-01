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
    OptGraph(ir::Graph* graph, RuntimePartitionInfo* info, RefitGraphInfo* refit_info, CudaArgs* args,
             CompileInfo* compile_set);
    ~OptGraph();

    ppl::common::RetCode DoOptimize(const utils::SharedResource&, CudaDevice*);

private:
    ppl::common::RetCode InitKernels();
    ppl::common::RetCode InitQuantization();
    ppl::common::RetCode UpdateDims(const utils::SharedResource& resource, CudaDevice* dev);
    ppl::common::RetCode FuseOperator(const utils::SharedResource& resource);
    ppl::common::RetCode AddBridgeKernels(const utils::SharedResource& resource);
    ppl::common::RetCode UpdateType();
    ppl::common::RetCode SelectAlgos(const utils::SharedResource&, CudaDevice*);
    ppl::common::RetCode LoadConstants(CudaDevice*);
    ppl::common::RetCode DeleteBridgeKernels();
    int32_t LastLegalNodeIndex();
    void UpdateTopologicalSort();

private:
    ir::Graph* graph_;
    RuntimePartitionInfo* info_;
    RefitGraphInfo* refit_info_;
    CudaArgs* args_;
    CompileInfo* compile_set_;
    std::set<nodeid_t> illegal_dims_;
    utils::GenericCpuDevice default_cpu_device_;
    std::map<edgeid_t, std::unique_ptr<TensorImpl>> tensor_impls_;
    std::vector<nodeid_t> sorted_node_ids_;
    std::function<EdgeObject*(edgeid_t, uint32_t)> acquire_tensor_func_;
};

}}} // namespace ppl::nn::cuda

#endif
