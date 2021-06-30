#ifndef _ST_HPC_PPL_NN_ENGINES_ENGINE_CONTEXT_OPTIONS_H_
#define _ST_HPC_PPL_NN_ENGINES_ENGINE_CONTEXT_OPTIONS_H_

#include "ppl/nn/runtime/policy_defs.h"

namespace ppl { namespace nn {

struct EngineContextOptions {
    MemoryManagementPolicy mm_policy = MM_BETTER_PERFORMANCE;
};

}} // namespace ppl::nn

#endif
