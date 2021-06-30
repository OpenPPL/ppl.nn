#ifndef _ST_HPC_PPL_NN_RUNTIME_POLICY_DEFS_H_
#define _ST_HPC_PPL_NN_RUNTIME_POLICY_DEFS_H_

namespace ppl { namespace nn {

enum MemoryManagementPolicy {
    /** better performance policy, will use more memory */
    MM_BETTER_PERFORMANCE = 0,

    /** less memory policy, may cause performance loss */
    MM_LESS_MEMORY = 1,
};

}} // namespace ppl::nn

#endif
