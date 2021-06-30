#ifndef _ST_HPC_PPL_NN_OPTIMIZERS_GRAPH_OPTIMIZER_H_
#define _ST_HPC_PPL_NN_OPTIMIZERS_GRAPH_OPTIMIZER_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/ir/graph.h"

namespace ppl { namespace nn {

class GraphOptimizer {
public:
    virtual ~GraphOptimizer() {}
    virtual ppl::common::RetCode Optimize(ir::Graph*) const = 0;
};

}} // namespace ppl::nn

#endif
