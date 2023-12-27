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

#ifndef _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_ENGINE_FACTORY_H_
#define _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_ENGINE_FACTORY_H_

#include "engine_options.h"
#include "ppl/nn/common/common.h"
#include "ppl/nn/engines/engine.h"
#include <cuda_runtime.h>

namespace ppl { namespace nn { namespace llm { namespace cuda {

struct PPLNN_PUBLIC DeviceOptions final {
    int device_id = 0;
    uint32_t mm_policy = MM_COMPACT;
    cudaStream_t stream = 0;
};

struct HostDeviceOptions final {};

class PPLNN_PUBLIC EngineFactory final {
public:
    static Engine* Create(const EngineOptions& options);
    static DeviceContext* CreateDeviceContext(const DeviceOptions&);
    static DeviceContext* CreateHostDeviceContext(const HostDeviceOptions&);
};

}}}} // namespace ppl::nn::llm::cuda

#endif
