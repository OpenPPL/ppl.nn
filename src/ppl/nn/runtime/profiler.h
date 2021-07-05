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

#ifndef _ST_HPC_PPL_NN_RUNTIME_PROFILER_H_
#define _ST_HPC_PPL_NN_RUNTIME_PROFILER_H_

#include "ppl/nn/runtime/kernel_impl.h"
#include "ppl/nn/runtime/runtime_internal_conf.h"
#include "ppl/nn/runtime/runtime_graph.h"
#include "ppl/nn/runtime/runtime_aux_info.h"

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
#include "ppl/nn/runtime/profiling_statistics.h"
#include <chrono>
#endif

namespace ppl { namespace nn {

class Profiler final {
public:
    void Init(const RuntimeInternalConf* conf, const RuntimeGraph* graph, const RuntimeAuxInfo* aux_info);

    bool IsProfilingEnabled() const {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
        return conf_->profiling_flag;
#else
        return false;
#endif
    }

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    void CollectStatistics(KernelImpl*);

public:
    void StartProfiling(nodeid_t max_node_id);
    ppl::common::RetCode GetProfilingStatistics(ProfilingStatistics*) const;
    void StopProfiling();

private:
    struct KernelExecInfo {
        uint32_t exec_count = 0;
        uint64_t exec_microseconds = 0;
    };

    std::vector<KernelExecInfo> nodeid2info_;
#endif

private:
    const RuntimeInternalConf* conf_;
    const RuntimeGraph* graph_;
    const RuntimeAuxInfo* aux_info_;
};

}} // namespace ppl::nn

#endif
