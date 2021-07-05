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

#include "ppl/nn/engines/cuda/engine_context.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/default_cuda_device.h"
#include "ppl/nn/engines/cuda/buffered_cuda_device.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode CudaEngineContext::Init(const EngineContextOptions& options) {
    if (options.mm_policy != MM_LESS_MEMORY) {
        LOG(WARNING) << "unsupported mm policy[" << options.mm_policy << "]. CudaEngine supports MM_LESS_MEMORY only."
                     << " mm policy will be MM_LESS_MEMORY.";
    }

    // TODO implement other options
    auto status = device_.Init(MM_LESS_MEMORY);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init BufferedCudaDevice failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
