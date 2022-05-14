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

#include "../lua_engine.h"
#include "ppl/nn/engines/riscv/options.h"
#include "ppl/nn/common/logger.h"
#include "luacpp/luacpp.h"
using namespace std;
using namespace luacpp;
using namespace ppl::common;

namespace ppl { namespace nn { namespace lua { namespace riscv {

void RegisterRiscvEngine(const shared_ptr<LuaState>& lstate, const shared_ptr<LuaTable>& l_riscv_module) {
    auto lclass = lstate->CreateClass<LuaEngine>()
        .DefMember("GetName",
                   [](const LuaEngine* engine) -> const char* {
                       return engine->ptr->GetName();
                   });
    l_riscv_module->Set("Engine", lclass);
}

}}}} // namespace ppl::nn::lua::riscv
