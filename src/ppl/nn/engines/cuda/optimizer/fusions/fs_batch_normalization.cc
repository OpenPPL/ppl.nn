// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "ppl/nn/engines/cuda/optimizer/fusions/fs_batch_normalization.h"

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/params/batch_normalization_extra_param.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

const bool BatchNormalizationFusion::CanFuse(ir::Node* nextnode) {
    if (nextnode->GetType().name == "Relu") {
        return true;
    }
    return false;
}

const RetCode BatchNormalizationFusion::FuseBatchWithNextNode(ir::Node* node, ir::Node* nextnode, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto connect_edge_id = node->GetOutput(0);

    for (uint32_t i = 0; i < nextnode->GetOutputCount(); ++i) {
        auto edge_id = nextnode->GetOutput(i);
        auto temp_edge = topo->GetEdgeById(edge_id);
        temp_edge->SetProducer(node->GetId());
        if (i == 0) {
            node->ReplaceOutput(connect_edge_id, edge_id);
        } else {
            node->AddOutput(edge_id);
        }
    }

    for (uint32_t i = 0; i < nextnode->GetInputCount(); ++i) {
        auto edge_id = nextnode->GetInput(i);
        if (edge_id == connect_edge_id || edge_id == INVALID_EDGEID) {
            continue;
        }
        ir::Edge* edge = topo->GetEdgeById(edge_id);
        edge->DelConsumer(nextnode->GetId());
        edge->AddConsumer(node->GetId());
        node->AddInput(edge_id);
    }

    topo->DelEdgeById(connect_edge_id);
    topo->DelNodeById(nextnode->GetId());
    return RC_SUCCESS;
}

const RetCode BatchNormalizationFusion::FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto node_id = node->GetId();
    auto opt_kernel = (CudaOptKernel*)(options.info->kernels[node_id].get());
    CudaBatchNormalizationParam* param = (CudaBatchNormalizationParam*)opt_kernel->GetParam();

    if (node->GetOutputCount() != 1) {
        return RC_UNSUPPORTED;
    }
    auto edge_id = node->GetOutput(0);
    auto edge = topo->GetEdgeById(edge_id);
    if (topo->GetOutput(edge->GetName()) != INVALID_EDGEID) { // Can not fuse an output edge
        return RC_UNSUPPORTED;
    }

    auto iter = topo->GetEdgeById(edge_id)->CreateConsumerIter();
    if (!iter.IsValid()) {
        return RC_UNSUPPORTED;
    }

    auto nextnode_id = iter.Get();
    auto nextnode = topo->GetNodeById(nextnode_id);

    iter.Forward();
    if (iter.IsValid()) { // Do not fuse if the edge has more than one consumer
        return RC_UNSUPPORTED;
    }

    if (CanFuse(nextnode)) {
        LOG(DEBUG) << "Fuse node[" << node->GetName() << "] and nextnode[" << nextnode->GetName() << "]";
        param->extra_param.has_relu = true;
        options.info->kernels.erase(nextnode_id);
        FuseBatchWithNextNode(node, nextnode, options);
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda