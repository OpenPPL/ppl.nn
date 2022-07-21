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
#include "ppl/nn/engines/cuda/plain_cuda_device.h"
#include "ppl/nn/engines/cuda/buffered_cuda_device.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode CudaEngineContext::Init(const EngineOptions& options) {
    if (options.mm_policy == MM_PLAIN) {
        auto dev = make_shared<PlainCudaDevice>();
        auto rc = dev->Init(options.device_id);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "init DefaultCudaDevice failed: " << GetRetCodeStr(rc);
            return rc;
        }
        device_ = dev;
    } else {
        auto dev = make_shared<BufferedCudaDevice>();
        auto rc = dev->Init(options.device_id, options.mm_policy);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "init BufferedCudaDevice failed: " << GetRetCodeStr(rc);
            return rc;
        }
        device_ = dev;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
