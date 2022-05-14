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

#include "ppl/nn/engines/cuda/engine_options.h"
#include "luacpp/luacpp.h"
#include <memory>
using namespace std;
using namespace luacpp;
using namespace ppl::nn::cuda;

namespace ppl { namespace nn { namespace lua { namespace cuda {

void RegisterEngineOptions(const shared_ptr<LuaState>& lstate, const shared_ptr<LuaTable>& l_cuda_module) {
    auto lclass = lstate->CreateClass<EngineOptions>()
        .DefConstructor()
        .DefMember<uint32_t>("device_id",
                             [](const EngineOptions* options) -> uint32_t {
                                 return options->device_id;
                             },
                             [](EngineOptions* options, uint32_t v) -> void {
                                 options->device_id = v;
                             })
        .DefMember<uint32_t>("mm_policy",
                             [](const EngineOptions* options) -> uint32_t {
                                 return options->mm_policy;
                             },
                             [](EngineOptions* options, uint32_t v) -> void {
                                 options->mm_policy = v;
                             });
    l_cuda_module->Set("EngineOptions", lclass);

    l_cuda_module->SetInteger("MM_COMPACT", MM_COMPACT);
    l_cuda_module->SetInteger("MM_BEST_FIT", MM_BEST_FIT);
}

}}}} // namespace ppl::nn::lua::cuda
