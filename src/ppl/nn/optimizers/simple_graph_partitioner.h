#ifndef _ST_HPC_PPL_NN_OPTIMIZERS_SIMPLE_GRAPH_PARTITIONER_H_
#define _ST_HPC_PPL_NN_OPTIMIZERS_SIMPLE_GRAPH_PARTITIONER_H_

#include "ppl/nn/ir/graph.h"
#include "ppl/nn/utils/shared_resource.h"
#include <vector>

namespace ppl { namespace nn {

class SimpleGraphPartitioner final {
public:
    ppl::common::RetCode Partition(utils::SharedResource*, ir::Graph*,
                                   std::vector<std::pair<EngineImpl*, std::vector<nodeid_t>>>*) const;
};

}} // namespace ppl::nn

#endif
