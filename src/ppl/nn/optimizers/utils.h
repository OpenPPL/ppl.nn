#ifndef _ST_HPC_PPL_NN_OPTIMIZERS_UTILS_H_
#define _ST_HPC_PPL_NN_OPTIMIZERS_UTILS_H_

#include "ppl/nn/utils/shared_resource.h"
#include "ppl/nn/ir/graph.h"
#include "ppl/nn/runtime/runtime_graph_info.h"

namespace ppl { namespace nn { namespace utils {

/**
   @brief optimize the compute graph `graph` and fill `info`
   @param resource global resources object
   @param graph that needed to be processed
   @param info fields needed
*/
ppl::common::RetCode ProcessGraph(utils::SharedResource* resource, ir::Graph* graph, RuntimeGraphInfo* info);

}}} // namespace ppl::nn::utils

#endif
