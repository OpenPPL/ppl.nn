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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_COMMON_THREADING_TOOLS_H_
#define __ST_PPL_KERNEL_ARM_SERVER_COMMON_THREADING_TOOLS_H_

#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server {

struct single_parallel_loop_config_t {
    int64_t depth_of_loop = 0;
    int64_t num_threads   = 1;
};

// A very naive version
inline single_parallel_loop_config_t select_single_parallel_loop(
    const int64_t* iter_of_loop,
    const int64_t loop_depth,
    const float omp_div_task_time_ratio,
    const uint8_t* forbid_mask = nullptr)
{
    const int64_t omp_max_thread_num = PPL_OMP_MAX_THREADS();

    int64_t task_cost_single_thread = 1;
    for (int64_t i = 0; i < loop_depth; i++) {
        task_cost_single_thread *= iter_of_loop[i];
    }

    int64_t task_count                 = 1;
    float min_total_time               = __FLT_MAX__;
    int64_t min_total_time_depth       = 0;
    int64_t min_total_time_num_threads = 1;
    for (int64_t i = 0; i < loop_depth; i++) {
        const int64_t thread_num              = min(iter_of_loop[i], omp_max_thread_num);
        const int64_t omp_create_thread_times = task_count * thread_num;
        const float task_time                 = (float)task_cost_single_thread / thread_num;
        const float total_time                = task_time + omp_create_thread_times * omp_div_task_time_ratio;
        if (forbid_mask == nullptr || forbid_mask[i] == 0) {
            if (total_time < min_total_time) {
                min_total_time             = total_time;
                min_total_time_depth       = i;
                min_total_time_num_threads = thread_num;
            }
        }
        task_count *= iter_of_loop[i];
    }

    single_parallel_loop_config_t config;
    config.depth_of_loop = min_total_time_depth;
    config.num_threads   = min_total_time_num_threads;

    return config;
}

inline single_parallel_loop_config_t select_single_parallel_loop(
    const std::vector<int64_t>& iter_of_loop,
    const float omp_div_task_time_ratio,
    const std::vector<uint8_t>& forbid_mask = std::vector<uint8_t>(0))
{
    return select_single_parallel_loop(iter_of_loop.data(), iter_of_loop.size(), omp_div_task_time_ratio, forbid_mask.data());
}

}}}; // namespace ppl::kernel::arm_server

#endif // __ST_PPL_KERNEL_ARM_SERVER_COMMON_THREADING_TOOLS_H_