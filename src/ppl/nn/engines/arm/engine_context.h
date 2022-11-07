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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_ENGINE_CONTEXT_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_ENGINE_CONTEXT_H_

#include "ppl/nn/engines/engine_context.h"
#include "ppl/nn/engines/arm/runtime_arm_device.h"
#include <string>

namespace ppl { namespace nn { namespace arm {

#define ARM_DEFAULT_ALIGNMENT 64u

class ArmEngineContext final : public EngineContext {
public:
    ArmEngineContext() {}

    ppl::common::RetCode Init(ppl::common::isa_t isa, uint32_t mm_policy);

    Device* GetDevice() const override {
        return device_.get();
    }
    const char* GetName() const override {
        return "arm";
    }

private:
    std::shared_ptr<ArmDevice> device_;

private:
    ArmEngineContext(const ArmEngineContext&) = delete;
    ArmEngineContext& operator=(const ArmEngineContext&) = delete;
};

}}} // namespace ppl::nn::arm

#endif
