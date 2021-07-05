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

#ifndef __ST_PPL_KERNEL_X86_COMMON_OMP_TOOLS_H_
#define __ST_PPL_KERNEL_X86_COMMON_OMP_TOOLS_H_

#include <vector>

#include "ppl/kernel/x86/common/general_include.h"
#include "ppl/common/sys.h"

namespace ppl { namespace kernel { namespace x86 {

/*
    mode:
        0: binding by core list
        1: binding by omp tid
*/
void set_omp_core_binding(const int32_t *cores, const int32_t num_cores, const int32_t mode);

int32_t get_omp_max_threads();

struct single_parallel_loop_config_t {
    int64_t depth_of_loop;
    int64_t num_threads;
};

// A very naive version
single_parallel_loop_config_t select_single_parallel_loop(
    const std::vector<int64_t> &iter_of_loop,
    const ppl::common::isa_t isa_flag,
    const uint64_t load_per_task,
    const uint64_t store_per_task,
    const uint64_t output_per_task,
    const uint64_t op_per_output);

// use forbid mask to indicate which dims cannot be paralleled
single_parallel_loop_config_t select_single_parallel_loop_with_mask(
    const std::vector<int64_t> &iter_of_loop,
    const std::vector<bool> &forbid_mask,
    const ppl::common::isa_t isa_flag,
    const uint64_t load_per_task,
    const uint64_t store_per_task,
    const uint64_t output_per_task,
    const uint64_t op_per_output);

}}}; // namespace ppl::kernel::x86

#endif
