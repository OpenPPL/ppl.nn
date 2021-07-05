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

#ifndef _ST_HPC_PPL_NN_ENGINES_COMMON_COMMON_KERNEL_IMPL_H_
#define _ST_HPC_PPL_NN_ENGINES_COMMON_COMMON_KERNEL_IMPL_H_

#include "ppl/nn/runtime/kernel_impl.h"
#include "ppl/nn/utils/cpu_timing_guard.h"

namespace ppl { namespace nn { namespace common {

class CommonKernelImpl : public KernelImpl {
public:
    CommonKernelImpl(const ir::Node* node) : KernelImpl(node) {}

    ppl::common::RetCode Execute(KernelExecContext* ctx) override final {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
        utils::CpuTimingGuard __timing_guard__(&begin_ts_, &end_ts_, ctx->IsProfilingEnabled());
#endif
        return DoExecute(ctx);
    }

protected:
    virtual ppl::common::RetCode DoExecute(KernelExecContext*) = 0;

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
public:
    uint64_t GetExecutionTime() const override final {
        auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end_ts_ - begin_ts_);
        return diff.count();
    }

private:
    std::chrono::time_point<std::chrono::system_clock> begin_ts_;
    std::chrono::time_point<std::chrono::system_clock> end_ts_;
#endif
};

}}} // namespace ppl::nn::common

#endif
