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

#include "../../engines/lua_engine.h"
#include "ppl/nn/models/pmx/runtime_builder_factory.h"
#include "lua_pmx_runtime_builder.h"
#include "luacpp/luacpp.h"
using namespace std;
using namespace luacpp;

namespace ppl { namespace nn { namespace lua {

void RegisterPmxRuntimeBuilderFactory(const shared_ptr<LuaState>& lstate, const shared_ptr<LuaTable>& lmodule) {
    auto builder_class = LuaClass<LuaPmxRuntimeBuilder>(lmodule->Get("RuntimeBuilder"));

    auto lclass = lstate->CreateClass<pmx::RuntimeBuilderFactory>()
        .DefStatic("Create", [builder_class, lstate]() -> LuaObject {
            auto builder = pmx::RuntimeBuilderFactory::Create();
            if (!builder) {
                return lstate->CreateNil();
            }
            return builder_class.CreateUserData(builder);
        });

    lmodule->Set("RuntimeBuilderFactory", lclass);
}

}}} // namespace ppl::nn::lua
