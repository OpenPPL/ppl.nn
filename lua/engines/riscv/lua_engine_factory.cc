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

#include "ppl/nn/engines/riscv/engine_factory.h"
#include "ppl/nn/common/logger.h"
#include "../lua_engine.h"
#include "luacpp/luacpp.h"
using namespace std;
using namespace luacpp;
using namespace ppl::nn::riscv;

namespace ppl { namespace nn { namespace lua { namespace riscv {

void RegisterEngineFactory(const shared_ptr<LuaState>& lstate, const shared_ptr<LuaTable>& l_riscv_module) {
    auto riscv_engine_class = LuaClass<LuaEngine>(l_riscv_module->Get("Engine"));
    auto lclass = lstate->CreateClass<EngineFactory>()
        .DefStatic("Create",
                   [riscv_engine_class, lstate](const EngineOptions* options) -> LuaObject {
                       auto engine = EngineFactory::Create(*options);
                       if (!engine) {
                           return lstate->CreateNil();
                       }
                       return riscv_engine_class.CreateInstance(engine);
                   });
    l_riscv_module->Set("EngineFactory", lclass);
}

}}}} // namespace ppl::nn::lua::riscv
