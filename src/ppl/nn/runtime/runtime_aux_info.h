#ifndef _ST_HPC_PPL_NN_RUNTIME_RUNTIME_AUX_INFO_H_
#define _ST_HPC_PPL_NN_RUNTIME_RUNTIME_AUX_INFO_H_

#include "ppl/nn/common/types.h"
#include "ppl/nn/runtime/runtime_graph_info.h"
#include <vector>

namespace ppl { namespace nn {

/**
   @class RuntimeAuxInfo
   @brief auxiliary info for runtime stage
*/
struct RuntimeAuxInfo final {
    /** node ids in topological order */
    std::vector<nodeid_t> sorted_nodes;
};

ppl::common::RetCode GenerateRuntimeAuxInfo(const RuntimeGraphInfo&, RuntimeAuxInfo*);

}} // namespace ppl::nn

#endif
