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

#include "ppl/nn/engines/cuda/optimizer/fusions/fs_conv.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/engines/cuda/params/conv_extra_param.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

const RetCode ConvFusion::FuseConvWithNextNode(ir::Node* node, ir::Node* nextnode, const OptKernelOptions& options) {
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

const bool ConvFusion::FuseTest(ir::Node* node, const OptKernelOptions& options,
                                std::function<ppl::common::RetCode(ir::Node*, const OptKernelOptions&)> canfuse) {
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto node_id = node->GetId();
    auto opt_kernel = (CudaOptKernel*)(options.info->kernels[node_id].get());
    CudaConvParam* param = (CudaConvParam*)opt_kernel->GetParam();

    auto edge_id = node->GetOutput(0);
    auto edge = topo->GetEdgeById(edge_id);
    if (topo->GetOutput(edge->GetName()) != INVALID_EDGEID) { // Can not fuse an output edge
        return false;
    }
    if (topo->GetEdgeById(edge_id)->CalcConsumerCount() != 1) { // Can not fuse multi-consumer edge
        return false;
    }

    auto nextnode_id = topo->GetEdgeById(edge_id)->CreateConsumerIter().Get(); // Get Output(0)
    auto nextnode = topo->GetNodeById(nextnode_id);

    if (canfuse(nextnode, options)) {
        LOG(DEBUG) << "Fuse node[" << node->GetName() << "] and nextnode[" << nextnode->GetName() << "]";
        param->extra_param.fuse_info.types.emplace_back(nextnode->GetType().name);
        param->extra_param.fuse_info.input_ind.emplace_back(node->GetInputCount());

        if (nextnode->GetType().name != "Clip") {
            auto next_kernel = (CudaOptKernel*)(options.info->kernels[nextnode_id].get());
            void* temp_param = nullptr;
            next_kernel->CopyParam(temp_param);
            param->extra_param.fuse_info.fuse_attrs.emplace_back(std::move(temp_param));
        } else {
            auto clip_param = new ClipParam();
            auto min_iter = data->constants.find(nextnode->GetInput(1));
            if (min_iter != data->constants.end()) {
                clip_param->min_val = *(float*)(min_iter->second.data.data());
            }
            auto max_iter = data->constants.find(nextnode->GetInput(2));
            if (max_iter != data->constants.end()) {
                clip_param->max_val = *(float*)(max_iter->second.data.data());
            }
            param->extra_param.fuse_info.fuse_attrs.emplace_back((void*)clip_param);
        }
        options.info->kernels.erase(nextnode_id);
        FuseConvWithNextNode(node, nextnode, options);
        return true;
    }
    return false;
}

const RetCode ConvFusion::FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) {
    FuseTest(node, options, CanFuseRelu);
    if (reliable) {
        if (FuseTest(node, options, CanFuseElementwise)) {
            FuseTest(node, options, CanFuseRelu);
        }
    }
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda