#ifndef _ST_HPC_PPL_NN_UTILS_SHARED_RESOURCE_H_
#define _ST_HPC_PPL_NN_UTILS_SHARED_RESOURCE_H_

#include "ppl/nn/engines/engine_impl.h"

namespace ppl { namespace nn { namespace utils {

struct SharedResource {
    std::vector<std::unique_ptr<EngineImpl>> engines;
};

}}} // namespace ppl::nn::utils

#endif
