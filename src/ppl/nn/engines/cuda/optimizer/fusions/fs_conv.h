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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_FUSIONS_FS_CONV_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_FUSIONS_FS_CONV_H_

#include "ppl/nn/engines/cuda/optimizer/fusions/fusion.h"

namespace ppl { namespace nn { namespace cuda {

class ConvFusion : public Fusion {
public:
    const ppl::common::RetCode FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) override;

private:
    const bool FuseTest(ir::Node* node, const OptKernelOptions& options,
                        std::function<ppl::common::RetCode(ir::Node*, const OptKernelOptions&)>);
    const ppl::common::RetCode FuseConvWithNextNode(ir::Node* node, ir::Node* nextnode,
                                                    const OptKernelOptions& options);

    static bool CanFuseRelu(ir::Node* nextnode, const OptKernelOptions& options) {
        // std::set<std::string> relu_fuse_op{"Relu", "Clip", "PRelu", "LeakyRelu", "Sigmoid"};
        std::set<std::string> relu_fuse_op{"Relu", "Clip"};
        if (relu_fuse_op.find(nextnode->GetType().name) != relu_fuse_op.end()) {
            if (nextnode->GetType().name == "PRelu") { // extra check for PRelu
                // slope must be an 1-d array or a scalar
                const TensorShape& shape1 = *options.tensors->find(nextnode->GetInput(0))->second->GetShape();
                const TensorShape& shape2 = *options.tensors->find(nextnode->GetInput(1))->second->GetShape();
                if (shape2.IsScalar() || (shape2.GetDimCount() == 1 && shape1.GetDim(1) == shape2.GetDim(0))) {
                    return true;
                }
                return false;
            }
            return true;
        }
        return false;
    }

    static bool CanFuseElementwise(ir::Node* nextnode, const OptKernelOptions& options) {
        std::set<std::string> elementwise_fuse_op{"Add", "Eltwise"};
        if (elementwise_fuse_op.find(nextnode->GetType().name) != elementwise_fuse_op.end()) {
            // two inputs must have same dims size for conv-add fusion
            const TensorShape& shape1 = *options.tensors->find(nextnode->GetInput(0))->second->GetShape();
            const TensorShape& shape2 = *options.tensors->find(nextnode->GetInput(1))->second->GetShape();
            if (shape1.GetDimCount() != shape2.GetDimCount()) {
                return false;
            }
            for (uint32_t i = 0; i < shape1.GetDimCount(); ++i) {
                if (shape1.GetDim(i) != shape2.GetDim(i)) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }
};

}}} // namespace ppl::nn::cuda

#endif
