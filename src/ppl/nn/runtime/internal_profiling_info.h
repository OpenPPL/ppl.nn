#ifndef _ST_HPC_PPL_NN_RUNTIME_INTERNAL_PROFILING_INFO_H_
#define _ST_HPC_PPL_NN_RUNTIME_INTERNAL_PROFILING_INFO_H_

namespace ppl { namespace nn {

struct InternalProfilingInfo final {
    uint64_t exec_microseconds = 0;
};

}}

#endif
