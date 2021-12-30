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

#include "ppl/nn/engines/cuda/optimizer/algos/algo_bridge.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

double BridgeAlgorithm::ExcuteTimer(const ir::Node* node, OptKernelOptions& options) {
    auto data = options.graph->data.get();
    auto preedge_id = node->GetInput(0);
    auto preshape = options.tensors->find(preedge_id)->second.get()->GetShape();
    double timer = 0.0;

    if (input_format_ != output_format_) {
        if (data->constants.find(preedge_id) != data->constants.end()) {
            return timer = 0.0;
        }

        return 1e-7 * preshape.GetElementsIncludingPadding();
    }
    return timer;
}

void BridgeAlgorithm::ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                                     dataformat_t input_format, dataformat_t output_format) {
    input_format_ = input_format;
    output_format_ = output_format;
    return;
}

}}} // namespace ppl::nn::cuda
