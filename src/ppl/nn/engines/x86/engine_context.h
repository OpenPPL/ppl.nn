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

#ifndef _ST_HPC_PPL_NN_ENGINES_X86_ENGINE_CONTEXT_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_ENGINE_CONTEXT_H_

#include "ppl/nn/engines/x86/x86_device.h"
#include "ppl/nn/engines/x86/kernel.h"
#include "ppl/nn/engines/x86/runtime_x86_device.h"
#include "ppl/nn/engines/engine_context.h"

namespace ppl { namespace nn { namespace x86 {

#define X86_DEFAULT_ALIGNMENT 64u

class X86EngineContext final : public EngineContext {
public:
    X86EngineContext(const std::string& name, ppl::common::isa_t isa, uint32_t mm_policy)
        : name_(name), device_(X86_DEFAULT_ALIGNMENT, isa, mm_policy) {}
    Device* GetDevice() override {
        return &device_;
    }

private:
    const std::string name_;
    RuntimeX86Device device_;
};

}}} // namespace ppl::nn::x86

#endif
