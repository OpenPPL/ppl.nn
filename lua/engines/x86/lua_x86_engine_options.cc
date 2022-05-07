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

namespace ppl { namespace nn { namespace lua {

void RegisterX86EngineOptions(const shared_ptr<LuaState>& lstate, const shared_ptr<LuaTable>& lmodule) {
    auto lclass = lstate->CreateClass<x86::EngineOptions>()
        .DefConstructor()
        .DefMember<uint32_t>("mm_policy",
                             [](const x86::EngineOptions* options) -> uint32_t {
                                 return options->mm_policy;
                             },
                             [](x86::EngineOptions* options, uint32_t v) -> void {
                                 options->mm_policy = v;
                             });
    lmodule->Set("EngineOptions", lclass);

    lmodule->SetInteger("MM_MRU", x86::MM_MRU);
    lmodule->SetInteger("MM_COMPACT", x86::MM_COMPACT);
}

}}}
