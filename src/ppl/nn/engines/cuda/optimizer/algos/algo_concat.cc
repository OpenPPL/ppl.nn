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

#include "ppl/nn/engines/cuda/optimizer/algos/algo_concat.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

double ConcatAlgorithm::ExcuteTimer(const ir::Node* node, OptKernelOptions& options) {
    const TensorShape& shape = *options.tensors->find(node->GetOutput(0))->second->GetShape();
    double timer = 1.0e-7 * shape.GetElementsIncludingPadding();
    return timer;
}

void ConcatAlgorithm::ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                                     dataformat_t input_format, dataformat_t output_format) {
    for (uint32_t i = 0; i < node->GetInputCount(); ++i) {
        auto edge_id = node->GetInput(i);
        if (edge_id == INVALID_EDGEID) {
            continue;
        }
        auto shape = tensors->find(edge_id)->second->GetShape();
        shape->SetDataFormat(input_format);
    }

    for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
        auto edge_id = node->GetOutput(i);
        auto shape = tensors->find(edge_id)->second->GetShape();
        shape->SetDataFormat(output_format);
    }
    return;
}

}}} // namespace ppl::nn::cuda
