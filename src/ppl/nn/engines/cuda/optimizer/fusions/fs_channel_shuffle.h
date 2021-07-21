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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_FUSIONS_FS_CHANNEL_SHUFFLE_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_FUSIONS_FS_CHANNEL_SHUFFLE_H_

#include "ppl/nn/engines/cuda/optimizer/fusions/fusion.h"

namespace ppl { namespace nn { namespace cuda {

class ChannelShuffleFusion : public Fusion {
public:
    const ppl::common::RetCode FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) override;

private:
    const bool CanFuse(ir::Node* node, const OptKernelOptions& options);
    const bool CanFuseFirstReshape(ir::Node* node, const OptKernelOptions& options);
    const bool CanFuseTranspose(ir::Node* node, const OptKernelOptions& options);
    const bool CanFuseSecondReshape(ir::Node* node, const OptKernelOptions& options);
    const ppl::common::RetCode FuseWithNextNodes(ir::Node* node, const OptKernelOptions& options);
};

}}} // namespace ppl::nn::cuda

#endif