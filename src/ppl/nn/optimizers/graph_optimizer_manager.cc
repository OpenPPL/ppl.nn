#include "ppl/nn/optimizers/graph_optimizer_manager.h"
#include "ppl/nn/common/logger.h"

#include "ppl/nn/optimizers/constant_node_optimizer.h"
#include "ppl/nn/optimizers/fuse_parallel_node_optimizer.h"
#include "ppl/nn/optimizers/fuse_bn_optimizer.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

#define REGISTER_OPTIMIZER(name, type) name2optimizer_.emplace(name, unique_ptr<GraphOptimizer>(new type()))

GraphOptimizerManager::GraphOptimizerManager() {
    REGISTER_OPTIMIZER("ConstantNodeOptimizer", ConstantNodeOptimizer);
    REGISTER_OPTIMIZER("FuseParallelNodeOptimizer", FuseParallelNodeOptimizer);
    REGISTER_OPTIMIZER("FuseBNOptimizer", FuseBNOptimizer);
}

RetCode GraphOptimizerManager::Process(ir::Graph* graph) const {
    for (auto x = name2optimizer_.begin(); x != name2optimizer_.end(); ++x) {
        auto status = x->second->Optimize(graph);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "optimizer[" << x->first << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

}} // namespace ppl::nn
