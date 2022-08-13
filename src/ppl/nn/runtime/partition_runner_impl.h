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

#ifndef _ST_HPC_PPL_NN_RUNTIME_PARTITION_RUNNER_IMPL_H_
#define _ST_HPC_PPL_NN_RUNTIME_PARTITION_RUNNER_IMPL_H_

#include "ppl/nn/runtime/partition_runner.h"
#include "ppl/nn/runtime/edge_object.h"
#include "ppl/nn/runtime/kernel_impl.h"
#include "ppl/nn/runtime/scheduler.h"
#include "ppl/nn/engines/engine_context.h"
#include "ppl/nn/ir/graph_topo.h"

namespace ppl { namespace nn {

class PartitionRunnerImpl final : public PartitionRunner {
public:
    ppl::common::RetCode Init(const std::shared_ptr<ir::GraphTopo>& topo, const std::vector<nodeid_t>& nodes,
                              std::vector<std::unique_ptr<EngineContext>>*, std::vector<EdgeObject*>* e2o,
                              std::vector<std::unique_ptr<KernelImpl>>* n2k);
    ppl::common::RetCode Run() override;

private:
    ppl::common::RetCode Sync();

private:
    std::unique_ptr<Scheduler> sched_;
    std::shared_ptr<ir::GraphTopo> topo_;
    std::vector<nodeid_t> sorted_nodes_;
    std::vector<nodeid_t> edge_last_consumer_;
    std::vector<std::unique_ptr<EngineContext>>* engctx_;
};

}} // namespace ppl::nn

#endif
