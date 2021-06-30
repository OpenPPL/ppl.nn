#ifndef _ST_HPC_PPL_NN_RUNTIME_PROFILING_STATISTICS_H_
#define _ST_HPC_PPL_NN_RUNTIME_PROFILING_STATISTICS_H_

#include "ppl/nn/common/common.h"
#include <vector>
#include <string>
#include <stdint.h>

namespace ppl { namespace nn {

struct PPLNN_PUBLIC KernelProfilingInfo final {
    std::string name;
    std::string domain;
    std::string type;
    uint64_t exec_microseconds;
    uint32_t exec_count;
};

struct PPLNN_PUBLIC ProfilingStatistics final {
    std::vector<KernelProfilingInfo> prof_info;
};

}} // namespace ppl::nn

#endif
