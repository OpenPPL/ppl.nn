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

#include "ppl/nn/engines/x86/engine_options.h"
#include "luacpp/luacpp.h"
#include <memory>
using namespace std;
using namespace luacpp;
using namespace ppl::nn::x86;

namespace ppl { namespace nn { namespace lua { namespace x86 {

void RegisterEngineOptions(const shared_ptr<LuaState>& lstate, const shared_ptr<LuaTable>& l_x86_module) {
    auto lclass = lstate->CreateClass<EngineOptions>()
        .DefConstructor()
        .DefMember<uint32_t>("mm_policy",
                             [](const EngineOptions* options) -> uint32_t {
                                 return options->mm_policy;
                             },
                             [](EngineOptions* options, uint32_t v) -> void {
                                 options->mm_policy = v;
                             });
    l_x86_module->Set("EngineOptions", lclass);

    l_x86_module->SetInteger("MM_MRU", MM_MRU);
    l_x86_module->SetInteger("MM_COMPACT", MM_COMPACT);
}

}}}} // namespace ppl::nn::lua::x86
