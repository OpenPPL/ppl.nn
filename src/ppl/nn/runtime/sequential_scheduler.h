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

#ifndef _ST_HPC_PPL_NN_RUNTIME_SEQUENTIAL_SCHEDULER_H_
#define _ST_HPC_PPL_NN_RUNTIME_SEQUENTIAL_SCHEDULER_H_

#include "ppl/nn/runtime/scheduler.h"
#include "ppl/common/object_pool.h"
#include "ppl/nn/runtime/tensor_sequence.h"

namespace ppl { namespace nn {

class SequentialScheduler final : public Scheduler {
public:
    ppl::common::RetCode Init(const ir::GraphTopo* topo, const RuntimeAuxInfo* aux_info, RuntimeGraph* g) override;
    ppl::common::RetCode Run(Profiler*) override;

private:
    const ir::GraphTopo* topo_;
    const RuntimeAuxInfo* aux_info_;
    RuntimeGraph* graph_;

    /** used to hold objects that are used during Run() */
    std::vector<EdgeObject*> edgeid2object_;

    /** used to accelerlate tensor allocations */
    ppl::common::ObjectPool<TensorImpl> tensor_pool_;

    /** used to accelerlate tensor sequence allocations */
    ppl::common::ObjectPool<TensorSequence> tensor_sequence_pool_;
};

}} // namespace ppl::nn

#endif
