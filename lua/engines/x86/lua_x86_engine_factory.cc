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

#ifdef PPLNN_USE_X86

#include "ppl/nn/engines/x86/engine_factory.h"
#include "ppl/nn/common/logger.h"
#include "luacpp.h"
#include "../lua_engine.h"
using namespace std;
using namespace luacpp;

namespace ppl { namespace nn { namespace lua {

void RegisterX86EngineFactory(const shared_ptr<LuaState>& lstate, const shared_ptr<LuaTable>& lmodule) {
    auto x86_engine_class = LuaClass<LuaEngine>(lmodule->Get("X86Engine"));
    auto lclass = lstate->CreateClass<X86EngineFactory>()
        .DefStatic("Create", [x86_engine_class, lstate](const X86EngineOptions* options) -> LuaObject {
            auto engine = X86EngineFactory::Create(*options);
            if (!engine) {
                return lstate->CreateNil();
            }
            return x86_engine_class.CreateUserData(engine);
        });
    lmodule->Set("X86EngineFactory", lclass);
}

}}}

#endif
