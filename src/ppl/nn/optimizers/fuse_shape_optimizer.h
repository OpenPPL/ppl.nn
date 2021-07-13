#ifndef _ST_HPC_PPL_NN_OPTIMIZERS_FUSE_SHAPE_OPTIMIZER_H_
#define _ST_HPC_PPL_NN_OPTIMIZERS_FUSE_SHAPE_OPTIMIZER_H_

#include "ppl/nn/optimizers/graph_optimizer.h"

namespace ppl { namespace nn {

class FuseShapeOptimizer : public GraphOptimizer {
public:
    virtual ~FuseShapeOptimizer() {}
    ppl::common::RetCode Optimize(ir::Graph*) const override;
};

}} // namespace ppl::nn

#endif
