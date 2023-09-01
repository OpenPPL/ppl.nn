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

#include "engine_context.h"
#include "plain_device.h"
#include "buffered_device.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace llm { namespace cuda {

RetCode LlmCudaEngineContext::Init(const EngineOptions& options, NcclParam* tensor_parallel_nccl_param) {
    if (options.mm_policy == MM_PLAIN) {
        device_.reset(new PlainDevice());
    } else if (options.mm_policy == MM_COMPACT) {
        device_.reset(new BufferedDevice());
    } else {
        LOG(ERROR) << "unsupported mm policy [" << options.mm_policy << "]";
        return RC_INVALID_VALUE;
    }

    auto rc = device_->Init(options.device_id, true, true, tensor_parallel_nccl_param);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "init device failed: " << GetRetCodeStr(rc);
        return rc;
    }

    return RC_SUCCESS;
}

}}}} // namespace ppl::nn::llm::cuda
