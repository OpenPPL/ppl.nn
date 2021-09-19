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

#include "lua_runtime.h"
#include "lua_runtime_builder.h"
#include "luacpp.h"
using namespace std;
using namespace luacpp;

namespace ppl { namespace nn { namespace lua {

void RegisterRuntimeBuilder(const shared_ptr<LuaState>& lstate, const shared_ptr<LuaTable>& lmodule) {
    auto runtime_class = LuaClass<LuaRuntime>(lmodule->Get("Runtime"));
    auto lclass = lstate->CreateClass<LuaRuntimeBuilder>()
        .DefMember("CreateRuntime", [runtime_class, lstate](LuaRuntimeBuilder* lbuilder) -> LuaObject {
            auto runtime = lbuilder->ptr->CreateRuntime();
            if (!runtime) {
                return lstate->CreateNil();
            }
            return runtime_class.CreateUserData(lbuilder->engines, runtime);
        });
    lmodule->Set("RuntimeBuilder", lclass);
}

}}}
