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
