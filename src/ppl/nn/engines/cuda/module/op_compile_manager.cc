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

#include "ppl/nn/engines/cuda/module/op_compile_manager.h"

namespace ppl { namespace nn { namespace cuda {

OpCompiler* OpCompilerManager::FindCompiler(const std::string& kernel_type) const {
    auto res = type2compiler_.find(kernel_type);
    if (res == type2compiler_.end()) {
        return nullptr;
    }
    return res->second;
}

OpCompilerManager::OpCompilerManager() {
    type2compiler_.emplace("Conv", &conv_);
    type2compiler_.emplace("Gemm", &gemm_);
    type2compiler_.emplace("ConvTranspose", &convtranspose_);
    type2compiler_.emplace("LSTM", &normal_);
    type2compiler_.emplace("MMCVModulatedDeformConv2d", &normal_);
}

}}} // namespace ppl::nn::cuda
