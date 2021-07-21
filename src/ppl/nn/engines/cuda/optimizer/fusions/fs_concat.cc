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

#include "ppl/nn/engines/cuda/optimizer/fusions/fs_concat.h"

#include "ppl/common/cuda/cuda_types.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/engines/cuda/params/conv_extra_param.h"
#include "ppl/nn/engines/cuda/params/concat_extra_param.h"

#define MAX_MASK_SIZE 32

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

const bool ConcatFusion::CanFuse(ir::Node* prenode, const OptKernelOptions& options, uint32_t offset_channel_size) {
    auto topo = options.graph->topo.get();
    auto shape = options.tensors->find(prenode->GetOutput(0))->second->GetShape();

    if (shape.GetDimCount() <= 1) {
        return false;
    }

    auto size = ppl::common::cuda::GetDataFormatChannelAlignment(shape.GetDataFormat());

    if (!size) {
        return false;
    }

    if (prenode->GetType().name != "Conv" || prenode->GetOutputCount() != 1 || offset_channel_size % size != 0 ||
        shape.GetDim(1) % size != 0) {
        return false;
    }

    auto edge = topo->GetEdgeById(prenode->GetOutput(0));
    if (edge->CalcConsumerCount() > 1) {
        return false;
    }

    return true;
}

const RetCode ConcatFusion::FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto node_id = node->GetId();
    auto opt_kernel = (CudaOptKernel*)(options.info->kernels[node_id].get());
    CudaConcatParam* param = (CudaConcatParam*)opt_kernel->GetParam();

    if (param->param.axis != 1 || !reliable) { // Only support fuse channel axis
        return RC_UNSUPPORTED;
    }

    uint32_t total_channel_size =
        options.tensors->find(node->GetOutput(0))->second->GetShape().GetDim(param->param.axis);
    uint32_t offset_channel_size = 0;

    for (uint32_t i = 0; i < node->GetInputCount(); ++i) {
        auto edge_id = node->GetInput(i);
        if (edge_id == INVALID_EDGEID) {
            continue;
        }
        auto prenode_id = topo->GetEdgeById(edge_id)->GetProducer();
        auto prenode = topo->GetNodeById(prenode_id);

        if (i < MAX_MASK_SIZE && CanFuse(prenode, options, offset_channel_size)) {
            LOG(DEBUG) << "Fuse lastnode[" << prenode->GetName() << "] and node[" << node->GetName()
                       << "] with channel offsets: " << offset_channel_size;
            param->extra_param.mask |= (1 << i);
            auto pre_kernel = (CudaOptKernel*)(options.info->kernels[prenode_id].get());
            CudaConvParam* conv_param = (CudaConvParam*)pre_kernel->GetParam();
            conv_param->extra_param.fuse_info.channel_offset = offset_channel_size;
            conv_param->extra_param.fuse_info.concat_edge_id = node->GetOutput(0);
            conv_param->extra_param.fuse_info.channel_size = total_channel_size;
        }

        auto preshape = options.tensors->find(edge_id)->second->GetShape();
        offset_channel_size += preshape.GetDim(param->param.axis);
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
