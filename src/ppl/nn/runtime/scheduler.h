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

#ifndef _ST_HPC_PPL_NN_RUNTIME_SCHEDULER_H_
#define _ST_HPC_PPL_NN_RUNTIME_SCHEDULER_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/runtime/runtime_graph_resource.h"
#include "ppl/nn/runtime/profiler.h"

namespace ppl { namespace nn {

class Scheduler {
public:
    struct Options final {
        Options(const ir::GraphTopo* t, const RuntimeAuxInfo* r, RuntimeGraphResource* g)
            : topo(t), aux_info(r), graph_resource(g) {}
        const ir::GraphTopo* topo;
        const RuntimeAuxInfo* aux_info;
        RuntimeGraphResource* graph_resource;
    };

public:
    virtual ~Scheduler() {}
    virtual ppl::common::RetCode Init(const Options&) = 0;
    virtual ppl::common::RetCode Run(const std::function<ppl::common::RetCode(KernelImpl*, KernelExecContext*)>&,
                                     Profiler*) = 0;
};

}} // namespace ppl::nn

#endif
