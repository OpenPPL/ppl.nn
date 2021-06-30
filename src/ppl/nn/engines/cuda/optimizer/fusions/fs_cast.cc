#include "ppl/nn/engines/cuda/optimizer/fusions/fs_cast.h"

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/common/logger.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

const bool CastFusion::CanFuse(ir::Node* node, ir::Node* prenode) {
    if (prenode->GetInputCount() == 1 && prenode->GetOutputCount() == 1 && node->GetType().name == "Cast" &&
        prenode->GetType().name == "Cast") {
        return true;
    }
    return false;
}

const RetCode CastFusion::FuseWithPreviousCast(ir::Node* node, ir::Node* prenode, OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto connect_edge_id = node->GetInput(0);
    auto connect_edge = topo->GetEdgeById(connect_edge_id);
    auto next_edge_id = node->GetOutput(0);
    auto next_edge = topo->GetEdgeById(next_edge_id);

    for (auto it = next_edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
        auto tempnode_id = it.Get();
        auto tempnode = topo->GetNodeById(tempnode_id);
        connect_edge->AddConsumer(tempnode_id);
        tempnode->ReplaceInput(next_edge_id, connect_edge_id);
    }

    connect_edge->DelConsumer(node->GetId());
    topo->DelEdgeById(next_edge->GetId());
    topo->DelNodeById(node->GetId());
    return RC_SUCCESS;
}

const RetCode CastFusion::FuseNode(ir::Node* node, bool reliable, OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto node_id = node->GetId();
    auto edge_id = node->GetInput(0);
    if (edge_id == INVALID_EDGEID) {
        return RC_UNSUPPORTED;
    }

    auto prenode_id = topo->GetEdgeById(edge_id)->GetProducer();
    if (prenode_id == INVALID_NODEID) {
        return RC_UNSUPPORTED;
    }

    auto prenode = topo->GetNodeById(prenode_id);
    if (node->GetInputCount() != 1 || node->GetOutputCount() != 1) {
        return RC_UNSUPPORTED;
    }

    auto edge = topo->GetEdgeById(node->GetOutput(0));
    if (topo->GetOutput(edge->GetName()) != INVALID_EDGEID) { // Can not fuse an output edge
        return RC_UNSUPPORTED;
    }

    if (CanFuse(node, prenode)) {
        LOG(DEBUG) << "Fuse cast node[" << node->GetName() << "] with prenode[" << prenode->GetName() << "]";
        options.info->kernels.erase(node_id);
        FuseWithPreviousCast(node, prenode, options);
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda