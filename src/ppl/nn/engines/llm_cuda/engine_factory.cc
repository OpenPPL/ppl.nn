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

#include "engine.h"
#include "plain_device.h"
#include "buffered_device.h"
#include "bestfit_buffered_device.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/llm_cuda/engine_factory.h"
#include "ppl/nn/utils/generic_cpu_device.h"
#include "ppl/common/retcode.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace llm { namespace cuda {

RetCode RegisterResourcesOnce();

Engine* EngineFactory::Create(const EngineOptions& options) {
    auto rc = RegisterResourcesOnce();
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "register llm cuda resources failed: " << GetRetCodeStr(rc);
        return nullptr;
    }

    auto engine = new LlmCudaEngine();
    if (engine) {
        rc = engine->Init(options);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "init llm cuda engine failed: " << GetRetCodeStr(rc);
            delete engine;
            return nullptr;
        }
    }

    return engine;
}

DeviceContext* EngineFactory::CreateDeviceContext(const DeviceOptions& options) {
    unique_ptr<LlmCudaDevice> dev;
    if (options.mm_policy == MM_PLAIN) {
        dev.reset(new PlainDevice(true));
    } else if (options.mm_policy == MM_BESTFIT) {
        dev.reset(new BestFitBufferedDevice());
    } else if (options.mm_policy == MM_COMPACT) {
        dev.reset(new BufferedDevice());
    } else {
        LOG(ERROR) << "unsupported mm policy [" << options.mm_policy << "]";
        return nullptr;
    }

    if (options.stream) {
        auto rc = dev->Init(options.device_id, true, nullptr, DeviceStreamFlag::SHARE, options.stream);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "init device failed: " << GetRetCodeStr(rc);
            return nullptr;
        }
    } else {
        auto rc = dev->Init(options.device_id, true, nullptr, DeviceStreamFlag::NEW);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "init device failed: " << GetRetCodeStr(rc);
            return nullptr;
        }
    }

    return dev.release();
}

DeviceContext* EngineFactory::CreateHostDeviceContext(const HostDeviceOptions&) {
    return new ppl::nn::utils::GenericCpuDevice();
}

}}}} // namespace ppl::nn::llm::cuda
