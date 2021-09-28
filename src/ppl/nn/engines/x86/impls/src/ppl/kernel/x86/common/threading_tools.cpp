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

#if defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#endif

#include "ppl/kernel/x86/common/threading_tools.h"
#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/common/log.h"
#include "ppl/common/sys.h"

namespace ppl { namespace kernel { namespace x86 {

void set_omp_core_binding(const int32_t *cores, const int32_t num_cores, const int32_t mode) {
    int32_t bmode = mode;
    if (num_cores < PPL_OMP_MAX_THREADS() || mode > 1 || mode < 0) {
        bmode = 1;
    }
#if defined(__linux__)
    PRAGMA_OMP_PARALLEL()
    {
        int32_t omp_tid = PPL_OMP_THREAD_ID();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (bmode == 0) {
            CPU_SET(cores[omp_tid], &cpuset);
        } else if (bmode == 1) {
            CPU_SET(omp_tid, &cpuset);
        }
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
            LOG(ERROR) << "Core binding failed";
        }
    }
#endif
}

int32_t get_omp_max_threads()
{
    return PPL_OMP_MAX_THREADS();
}
// A very naive version
single_parallel_loop_config_t select_single_parallel_loop(
    const std::vector<int64_t> &iter_of_loop,
    const ppl::common::isa_t isa_flag,
    const uint64_t load_per_task,
    const uint64_t store_per_task,
    const uint64_t output_per_task,
    const uint64_t op_per_output)
{
    const int64_t omp_max_threads = PPL_OMP_MAX_THREADS();
    if (iter_of_loop.size() == 0 || omp_max_threads == 1) {
        return {0, 1};
    }

    int64_t data_lane = 8; // for 64-bit system
    if (isa_flag >= ppl::common::ISA_X86_AVX512) {
        data_lane = 64;
    } else if (isa_flag >= ppl::common::ISA_X86_AVX) {
        data_lane = 32;
    } else if (isa_flag >= ppl::common::ISA_X86_SSE) {
        data_lane = 16;
    }

    // skylake, assume all overlap
    const int64_t inst_per_task =
        max<int64_t>(load_per_task / data_lane, 1) +
        max<int64_t>(store_per_task / data_lane, 1) +
        max<int64_t>(output_per_task / data_lane, 1) * op_per_output;

    std::vector<int64_t> task_of_iter(iter_of_loop.size());
    {
        int64_t depth = (int64_t)task_of_iter.size() - 1;
        int64_t tasks = 1;
        for (; depth >= 0; --depth) {
            if (iter_of_loop[depth] == 0) {
                return {0, 1};
            }
            tasks *= iter_of_loop[depth];
            task_of_iter[depth] = tasks;
        }
    }

    // is it a reasonable initial value?
    const uint64_t l2_inst      = 128 * 1024;
    int64_t max_thread_of_depth = 1;
    int64_t max_depth           = 0;
    for (int64_t depth = 0; depth < (int64_t)iter_of_loop.size(); ++depth) {
        const int64_t min_iter_per_thread = div_up(l2_inst, (task_of_iter[depth] * inst_per_task));
        const int64_t thread_of_depth     = min<int64_t>(div_up(iter_of_loop[depth], min_iter_per_thread), omp_max_threads);
        if (thread_of_depth > max_thread_of_depth) {
            max_thread_of_depth = thread_of_depth;
            max_depth           = depth;
        }
    }

    return {max_depth, max_thread_of_depth};
}

// use forbid mask to indicate which dims cannot be paralleled
single_parallel_loop_config_t select_single_parallel_loop_with_mask(
    const std::vector<int64_t> &iter_of_loop,
    const std::vector<bool> &forbid_mask,
    const ppl::common::isa_t isa_flag,
    const uint64_t load_per_task,
    const uint64_t store_per_task,
    const uint64_t output_per_task,
    const uint64_t op_per_output)
{
    const int64_t omp_max_threads = PPL_OMP_MAX_THREADS();
    if (iter_of_loop.size() == 0 || omp_max_threads == 1) {
        return {0, 1};
    }

    int64_t data_lane = 1;
    if (isa_flag >= ppl::common::ISA_X86_AVX512) {
        data_lane = 64;
    } else if (isa_flag >= ppl::common::ISA_X86_AVX) {
        data_lane = 32;
    } else if (isa_flag >= ppl::common::ISA_X86_SSE) {
        data_lane = 16;
    }

    // skylake, assume all overlap
    const int64_t inst_per_task =
        max<int64_t>(load_per_task / data_lane, 1) +
        max<int64_t>(store_per_task / data_lane, 1) +
        max<int64_t>(output_per_task / data_lane, 1) * op_per_output;

    std::vector<int64_t> task_of_iter(iter_of_loop.size());
    {
        int64_t depth = (int64_t)task_of_iter.size() - 1;
        int64_t tasks = 1;
        for (; depth >= 0; --depth) {
            tasks *= iter_of_loop[depth];
            task_of_iter[depth] = tasks;
        }
    }

    // is it a reasonable initial value?
    const uint64_t l2_inst      = 128 * 1024;
    int64_t max_thread_of_depth = 1;
    int64_t max_depth           = 0;
    for (int64_t depth = 0; depth < (int64_t)iter_of_loop.size(); ++depth) {
        if (depth < (int64_t)forbid_mask.size() && forbid_mask[depth] == true) { // skip when this dim cannot be paralleled
            continue;
        }
        const int64_t min_iter_per_thread = div_up(l2_inst, (task_of_iter[depth] * inst_per_task));
        const int64_t thread_of_depth     = min<int64_t>(div_up(iter_of_loop[depth], min_iter_per_thread), omp_max_threads);
        if (thread_of_depth > max_thread_of_depth) {
            max_thread_of_depth = thread_of_depth;
            max_depth           = depth;
        }
    }

    return {max_depth, max_thread_of_depth};
}

}}}; // namespace ppl::kernel::x86
