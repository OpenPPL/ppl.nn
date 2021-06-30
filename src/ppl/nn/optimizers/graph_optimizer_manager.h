#ifndef _ST_HPC_PPL_NN_OPTIMIZERS_GRAPH_OPTIMIZER_MANAGER_H_
#define _ST_HPC_PPL_NN_OPTIMIZERS_GRAPH_OPTIMIZER_MANAGER_H_

#include "ppl/nn/optimizers/graph_optimizer.h"
#include <map>

namespace ppl { namespace nn {

class GraphOptimizerManager final {
public:
    GraphOptimizerManager();

    /** @brief perform optimizations */
    ppl::common::RetCode Process(ir::Graph*) const;

private:
    std::map<std::string, std::unique_ptr<GraphOptimizer>> name2optimizer_;
};

}} // namespace ppl::nn

#endif
