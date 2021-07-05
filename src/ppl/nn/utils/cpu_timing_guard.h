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

#ifndef _ST_HPC_PPL_NN_UTILS_CPU_TIMING_GUARD_H_
#define _ST_HPC_PPL_NN_UTILS_CPU_TIMING_GUARD_H_

#include <chrono>

namespace ppl { namespace nn { namespace utils {

class CpuTimingGuard final {
public:
    CpuTimingGuard(std::chrono::time_point<std::chrono::system_clock>* begin_ts,
                   std::chrono::time_point<std::chrono::system_clock>* end_ts, bool is_profiling_enabled)
        : is_profiling_enabled_(is_profiling_enabled), end_ts_(end_ts) {
        if (is_profiling_enabled) {
            *begin_ts = std::chrono::system_clock::now();
        }
    }
    ~CpuTimingGuard() {
        if (is_profiling_enabled_) {
            *end_ts_ = std::chrono::system_clock::now();
        }
    }

private:
    bool is_profiling_enabled_;
    std::chrono::time_point<std::chrono::system_clock>* end_ts_;
};

}}} // namespace ppl::nn::utils

#endif
