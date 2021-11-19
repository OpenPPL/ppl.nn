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

#include "ppl/nn/engines/cuda/optimizer/fusions/fs_filter_manager.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

Fusion* FsFilterManager::FindFusion(const std::string& kernel_type) const {
    auto ref = type2fusion_.find(kernel_type);
    if (ref == type2fusion_.end()) {
        return nullptr;
    }
    return ref->second;
}

FsFilterManager::FsFilterManager() {
    type2fusion_.emplace("AveragePool", &averagepool_fs_);
    type2fusion_.emplace("Concat", &concat_fs_);
    type2fusion_.emplace("Conv", &conv_fs_);
    type2fusion_.emplace("Gemm", &gemm_fs_);
    type2fusion_.emplace("Reshape", &channel_shuffle_fs_);
    type2fusion_.emplace("Softmax", &softmax_fs_);
}

}}} // namespace ppl::nn::cuda
