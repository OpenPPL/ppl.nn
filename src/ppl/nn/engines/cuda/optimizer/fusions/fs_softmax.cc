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

#include "ppl/nn/engines/cuda/optimizer/fusions/fs_softmax.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/params/onnx/transpose_param.h"
#include "ppl/nn/params/onnx/softmax_param.h"

#include <vector>

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

const RetCode SoftmaxFusion::FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto pre_edge = topo->GetEdgeById(node->GetInput(0));
    auto post_edge = topo->GetEdgeById(node->GetOutput(0));

    if (pre_edge->GetProducer() == INVALID_NODEID || post_edge->CalcConsumerCount() == 0) {
        return RC_UNSUPPORTED;
    }

    auto pre_node = topo->GetNodeById(pre_edge->GetProducer());
    auto post_node = topo->GetNodeById(post_edge->CreateConsumerIter().Get());
    
    if (node->GetInputCount() != 1 || node->GetOutputCount() != 1) {
        return RC_UNSUPPORTED;
    }
    if (pre_edge->CalcConsumerCount() != 1 || post_edge->CalcConsumerCount() != 1) {
        return RC_UNSUPPORTED;
    }
    if (pre_node->GetType().name != "Transpose" || post_node->GetType().name != "Transpose") {
        return RC_UNSUPPORTED;
    }

    auto kernel = (CudaOptKernel*)(options.info->kernels.find(node->GetId())->second.get());
    auto pre_kernel = (CudaOptKernel*)(options.info->kernels.find(pre_node->GetId())->second.get());
    auto post_kernel = (CudaOptKernel*)(options.info->kernels.find(post_node->GetId())->second.get());

    auto softmax_param = (ppl::nn::common::SoftmaxParam*)(kernel->GetParam());
    if (softmax_param->axis != 3) {
        return RC_UNSUPPORTED;
    }
    std::vector<int32_t> perm = {0,3,2,1};
    auto pre_transpose_param = (ppl::nn::common::TransposeParam*)(pre_kernel->GetParam());
    if (pre_transpose_param->perm != perm) {
        return RC_UNSUPPORTED;
    }
    auto post_transpose_param = (ppl::nn::common::TransposeParam*)(post_kernel->GetParam());
    if (post_transpose_param->perm != perm) {
        return RC_UNSUPPORTED;
    }

    LOG(DEBUG) << "Give a better layout for node[" << node->GetName() << "].";
    pre_transpose_param->perm = std::vector<int32_t>{0,2,3,1};
    post_transpose_param->perm = std::vector<int32_t>{0,3,1,2};

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
