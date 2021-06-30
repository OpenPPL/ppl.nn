#ifndef _ST_HPC_PPL_NN_RUNTIME_RUNTIME_PARTITION_INFO_H_
#define _ST_HPC_PPL_NN_RUNTIME_RUNTIME_PARTITION_INFO_H_

#include "ppl/nn/runtime/opt_kernel.h"
#include "ppl/nn/runtime/runtime_constant_info.h"
#include <map>
#include <memory>

namespace ppl { namespace nn {

struct RuntimePartitionInfo {
    std::map<edgeid_t, RuntimeConstantInfo> constants;
    std::map<nodeid_t, std::unique_ptr<OptKernel>> kernels;
};

}} // namespace ppl::nn

#endif
