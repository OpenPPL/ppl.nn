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

#include "ppl/nn/runtime/profiler.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

void Profiler::Init(const vector<unique_ptr<KernelImpl>>* n2k, const RuntimeAuxInfo* aux_info) {
    nodeid2kernel_ = n2k;
    aux_info_ = aux_info;
}

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
void Profiler::CollectStatistics(KernelImpl* kernel) {
    auto info = &nodeid2info_[kernel->GetNode()->GetId()];
    InternalProfilingInfo intern_info;
    kernel->GetProfilingInfo(&intern_info);
    info->exec_microseconds += intern_info.exec_microseconds;
    ++info->exec_count;
}

void Profiler::StartProfiling(nodeid_t max_node_id) {
    nodeid2info_.resize(max_node_id);
}

RetCode Profiler::GetProfilingStatistics(ProfilingStatistics* stat) const {
    stat->prof_info.reserve(aux_info_->sorted_nodes.size());
    for (auto x = aux_info_->sorted_nodes.begin(); x != aux_info_->sorted_nodes.end(); ++x) {
        auto nid = *x;
        auto& info = nodeid2info_[nid];
        auto kernel = nodeid2kernel_->at(nid).get();

        KernelProfilingInfo kernel_prof_info;
        kernel_prof_info.name = kernel->GetName();
        auto& op_type = kernel->GetType();
        kernel_prof_info.domain = op_type.domain;
        kernel_prof_info.type = op_type.name;
        kernel_prof_info.exec_microseconds = info.exec_microseconds;
        kernel_prof_info.exec_count = info.exec_count;
        stat->prof_info.emplace_back(std::move(kernel_prof_info));
    }

    return RC_SUCCESS;
}

void Profiler::StopProfiling() {
    nodeid2info_.clear();
}
#endif

}} // namespace ppl::nn
