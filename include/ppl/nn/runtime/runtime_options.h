#ifndef _ST_HPC_PPL_NN_RUNTIME_RUNTIME_OPTIONS_H_
#define _ST_HPC_PPL_NN_RUNTIME_RUNTIME_OPTIONS_H_

#include "ppl/nn/runtime/policy_defs.h"

namespace ppl { namespace nn {

struct PPLNN_PUBLIC RuntimeOptions final {
    MemoryManagementPolicy mm_policy = MM_LESS_MEMORY;
};

}} // namespace ppl::nn

#endif
