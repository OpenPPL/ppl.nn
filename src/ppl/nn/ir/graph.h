#ifndef _ST_HPC_PPL_NN_IR_GRAPH_H_
#define _ST_HPC_PPL_NN_IR_GRAPH_H_

#include "ppl/nn/ir/graph_topo.h"
#include "ppl/nn/ir/graph_data.h"
#include <memory>

namespace ppl { namespace nn { namespace ir {

struct Graph final {
    std::shared_ptr<GraphTopo> topo;
    std::shared_ptr<GraphData> data;
};

}}} // namespace ppl::nn::ir

#endif
