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

#include "ppl/nn/engines/cuda/optimizer/fusions/fs_channel_shuffle.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/params/onnx/transpose_param.h"
#include "ppl/nn/params/ppl/channel_shuffle_param.h"
#include "ppl/nn/params/ppl/shape_operation_param.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

const bool ChannelShuffleFusion::CanFuseFirstReshape(ir::Node* node, const OptKernelOptions& options) {
    if (node->GetType().name != "Reshape") {
        return false;
    }
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto shape_edge_id = node->GetInput(1);
    auto constants_pair = data->constants.find(shape_edge_id);

    if (constants_pair != data->constants.end()) {
        auto dims = data->shapes.find(shape_edge_id)->second.dims;
        if (dims.size() != 1 || dims[0] != 5) {
            return false;
        }
        auto shape = (int64_t*)constants_pair->second.data.data();
        if (shape[1] != 2) {
            return false;
        }
        return true;
    }

    auto shape_node_id = topo->GetEdgeById(shape_edge_id)->GetProducer();
    auto shape_node = topo->GetNodeById(shape_node_id);
    if (shape_node->GetType().name != "Shape") {
        return false;
    }

    auto attr_pair = data->attrs.find(shape_node_id);
    if (attr_pair != data->attrs.end()) {
        auto param = (const ppl::nn::common::PPLShapeOperationParam*)(attr_pair->second.get());
        auto matrix = param->alpha.find(shape_edge_id)->second;
        if (matrix.numerator[1][matrix.MAXDIMSIZE] / matrix.denominator[1][matrix.MAXDIMSIZE] != 2) {
            return false;
        }
        return true;
    }
    return false;
}

const bool ChannelShuffleFusion::CanFuseTranspose(ir::Node* node, const OptKernelOptions& options) {
    if (node->GetType().name != "Transpose") {
        return false;
    }
    auto data = options.graph->data.get();
    auto attr_pair = data->attrs.find(node->GetId());

    if (attr_pair != data->attrs.end()) {
        auto param = (const ppl::nn::common::TransposeParam*)(attr_pair->second.get());
        std::vector<int32_t> shape{0, 2, 1, 3, 4};
        if (param->perm == shape) {
            return true;
        }
    }
    return false;
}

const bool ChannelShuffleFusion::CanFuseSecondReshape(ir::Node* node, const OptKernelOptions& options) {
    if (node->GetType().name != "Reshape") {
        return false;
    }
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto shape_edge_id = node->GetInput(1);
    auto constants_pair = data->constants.find(shape_edge_id);

    if (constants_pair != data->constants.end()) {
        auto dims = data->shapes.find(shape_edge_id)->second.dims;
        if (dims.size() != 1 || dims[0] != 4) {
            return false;
        }
        return true;
    }

    auto shape_node_id = topo->GetEdgeById(shape_edge_id)->GetProducer();
    auto attr_pair = data->attrs.find(shape_node_id);
    if (attr_pair != data->attrs.end()) {
        auto param = (const ppl::nn::common::PPLShapeOperationParam*)(attr_pair->second.get());
        auto matrix = param->alpha.find(shape_edge_id)->second;
        if (matrix.numerator[1][matrix.MAXDIMSIZE] / matrix.denominator[1][matrix.MAXDIMSIZE] != -1) {
            return false;
        }
        return true;
    }
    return false;
}

const bool ChannelShuffleFusion::CanFuse(ir::Node* node, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    for (uint32_t i = 0; i < 3; ++i) {
        switch (i) // TODO use function vector
        {
            case 0:
                if (!CanFuseFirstReshape(node, options)) {
                    return false;
                }
                break;
            case 1:
                if (!CanFuseTranspose(node, options)) {
                    return false;
                }
                break;
            case 2:
                if (!CanFuseSecondReshape(node, options)) {
                    return false;
                }
                break;
        }

        auto edge_id = node->GetOutput(0);
        auto edge = topo->GetEdgeById(edge_id);
        if (topo->GetOutput(edge->GetName()) != INVALID_EDGEID) { // Can not fuse an output edge
            return false;
        }
        if (i < 2 && topo->GetEdgeById(edge_id)->CalcConsumerCount() != 1) { // Can not fuse multi-consumer edge
            return false;
        }
        auto next_node_id = topo->GetEdgeById(edge_id)->CreateConsumerIter().Get(); // Get Output(0)
        node = topo->GetNodeById(next_node_id);
    }
    return true;
}

const bool ChannelShuffleFusion::CanFuseUpAndDown(ir::Node* node, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto pre_edge = topo->GetEdgeById(node->GetInput(0));
    auto post_edge = topo->GetEdgeById(node->GetOutput(0));
    auto pre_node = topo->GetNodeById(pre_edge->GetProducer());
    auto post_node = topo->GetNodeById(post_edge->CreateConsumerIter().Get());

    if (topo->GetOutput(pre_edge->GetName()) != INVALID_EDGEID ||
        topo->GetOutput(post_edge->GetName()) != INVALID_EDGEID) { // Can not fuse an output edge
        return false;
    }

    if (pre_node->GetType().name != "Concat" || pre_node->GetOutputCount() != 1) {
        return false;
    }

    if (post_node->GetType().name != "Split" || post_node->GetInputCount() != 1) {
        return false;
    }

    return true;
}

const RetCode ChannelShuffleFusion::FuseWithNextNodes(ir::Node* node, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto connect_edge_id = node->GetOutput(0);
    auto next_node_id = topo->GetEdgeById(connect_edge_id)->CreateConsumerIter().Get(); // Get Output(0)
    auto next_node = topo->GetNodeById(next_node_id);

    for (uint32_t i = 0; i < next_node->GetOutputCount(); ++i) {
        auto edge_id = next_node->GetOutput(i);
        auto temp_edge = topo->GetEdgeById(edge_id);
        temp_edge->SetProducer(node->GetId());
        if (i == 0) {
            node->ReplaceOutput(connect_edge_id, edge_id);
        } else {
            node->AddOutput(edge_id);
        }
    }

    for (uint32_t i = 0; i < next_node->GetInputCount(); ++i) {
        auto edge_id = next_node->GetInput(i);
        if (edge_id == connect_edge_id || edge_id == INVALID_EDGEID) {
            continue;
        }
        ir::Edge* edge = topo->GetEdgeById(edge_id);
        edge->DelConsumer(next_node->GetId());
        edge->AddConsumer(node->GetId());
        node->AddInput(edge_id);
    }

    topo->DelEdgeById(connect_edge_id);
    topo->DelNodeById(next_node->GetId());
    options.info->kernels.erase(next_node_id);
    return RC_SUCCESS;
}

const RetCode ChannelShuffleFusion::FuseWithLastNodes(ir::Node* next_node, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto next_node_id = next_node->GetId();
    auto connect_edge_id = next_node->GetInput(0);
    auto node_id = topo->GetEdgeById(connect_edge_id)->GetProducer(); // Get Output(0)
    auto node = topo->GetNodeById(node_id);
    auto shape_edge = topo->GetEdgeById(next_node->GetInput(1));
    auto shape_node = topo->GetNodeById(shape_edge->GetProducer());

    for (uint32_t i = 0; i < next_node->GetInputCount(); ++i) {
        auto edge_id = next_node->GetInput(i);
        auto edge = topo->GetEdgeById(edge_id);
        edge->DelConsumer(next_node_id);
        if (edge_id == connect_edge_id || edge->CalcConsumerCount() == 0) {
            topo->DelEdgeById(edge_id);
        }
    }

    for (uint32_t i = 0; i < next_node->GetOutputCount(); ++i) {
        auto edge_id = next_node->GetOutput(i);
        auto edge = topo->GetEdgeById(edge_id);
        node->ReplaceOutput(node->GetOutput(i), edge_id);
        edge->SetProducer(node_id);
    }

    topo->DelNodeById(next_node_id);
    if (shape_node) {
        options.info->kernels.erase(shape_node->GetId());
        topo->DelNodeById(shape_node->GetId());
    }
    return RC_SUCCESS;
}

const RetCode ChannelShuffleFusion::FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    if (CanFuse(node, options)) {
        LOG(DEBUG) << "Fuse node[" << node->GetName() << "] into channel shuffle";
        std::string node_name = "ChannelShuffle_" + node->GetName();
        options.info->kernels.erase(node->GetId());
        for (uint32_t i = 0; i < 2; ++i) {
            FuseWithNextNodes(node, options);
        }

        if (CanFuseUpAndDown(node, options)) {
            auto pre_edge = topo->GetEdgeById(node->GetInput(0));
            auto post_edge = topo->GetEdgeById(node->GetOutput(0));
            auto pre_node = topo->GetNodeById(pre_edge->GetProducer());
            auto post_node = topo->GetNodeById(post_edge->CreateConsumerIter().Get());

            LOG(DEBUG) << "Fuse pre_node[" << pre_node->GetName() << "] and post_node[" << post_node->GetName()
                       << "] into channel shuffle";
            FuseWithLastNodes(node, options);
            node = pre_node;
            options.info->kernels.erase(node->GetId());
            FuseWithNextNodes(node, options);
        }

        node->SetType(ir::Node::Type("ppl", "ChannelShuffle", 1));
        node->SetName(node_name);
        auto creator = OptKernelCreatorManager::Instance()->Find(node->GetType().domain, node->GetType().name,
                                                                 node->GetType().version);
        if (!creator) {
            LOG(ERROR) << "Cannot find creator for channel shuffle kernel";
            return RC_UNSUPPORTED;
        }

        auto opt_kernel = unique_ptr<CudaOptKernel>(creator(node));
        if (!opt_kernel) {
            LOG(ERROR) << "create Kernel failed: oom";
            return RC_UNSUPPORTED;
        }
        auto param = (ppl::nn::common::ChannelShuffleParam*)opt_kernel->GetParam();
        if (param == nullptr) {
            LOG(ERROR) << "Can not find param.";
            return RC_NOT_FOUND;
        }
        param->group = 2;
        opt_kernel->Init(options);
        options.info->kernels.emplace(node->GetId(), std::move(opt_kernel));
    }
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
