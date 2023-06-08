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

#include "ppl/common/retcode.h"
#include "ppl/common/log.h"
#include "ppl/nn/engines/cuda/utils.h"
#include "ppl/nn/engines/cuda/graph_scheduler.h"
#include "ppl/nn/engines/cuda/engine.h"
#include "ppl/nn/runtime/scheduler.h"
#include "ppl/nn/runtime/options.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {
RetCode SetGraphScheduler(Runtime* runtime, Engine* engine) {
    if ((!engine) || strcmp(engine->GetName(), "cuda")) {
        return RC_INVALID_VALUE;
    }
    auto sched = new cuda::CudaGraphScheduler();
    auto num = runtime->GetDeviceContextCount();
    for (uint32_t i = 0; i < num; ++i) {
        auto device = runtime->GetDeviceContext(i);
        if (strcmp(device->GetType(), "cuda")) {
            continue;
        }
        sched->GraphRunnerAddDevice(static_cast<cuda::CudaDevice*>(device));
    }
    sched->GraphRunnerAddDevice(static_cast<cuda::CudaEngine*>(engine)->GetDevice());
    return runtime->Configure(RUNTIME_CONF_SET_SCHEDULER, sched);
}
}}} // namespace ppl::nn::cuda