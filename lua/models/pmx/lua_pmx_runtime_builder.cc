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

#include "../../runtime/lua_runtime.h"
#include "../../engines/lua_engine.h"
#include "lua_pmx_runtime_builder.h"
#include "luacpp/luacpp.h"
using namespace std;
using namespace luacpp;
using namespace ppl::common;

namespace ppl { namespace nn { namespace lua {

void RegisterPmxRuntimeBuilder(const shared_ptr<LuaState>& lstate, const shared_ptr<LuaTable>& lmodule,
                               const shared_ptr<LuaTable>& l_pmx_module) {
    auto runtime_class = LuaClass<LuaRuntime>(lmodule->Get("Runtime"));
    auto lclass = lstate->CreateClass<LuaPmxRuntimeBuilder>()
        .DefMember("InitFromFile",
                   [](LuaPmxRuntimeBuilder* lbuilder, const char* model_file, const LuaTable& engines) -> RetCode {
                       vector<shared_ptr<Engine>> engine_list;
                       engines.ForEach([&engine_list](uint32_t, const LuaObject& value) -> bool {
                           engine_list.push_back(LuaUserData(value).Get<LuaEngine>()->ptr);
                           return true;
                       });

                       vector<Engine*> engine_ptrs(engine_list.size());
                       for (uint32_t i = 0; i < engine_list.size(); ++i) {
                           engine_ptrs[i] = engine_list[i].get();
                       }

                       lbuilder->engines = std::move(engine_list);
                       return lbuilder->ptr->Init(model_file, engine_ptrs.data(), engine_ptrs.size());
                   })
        .DefMember("Preprocess",
                   [](LuaPmxRuntimeBuilder* lbuilder) -> RetCode {
                       return lbuilder->ptr->Preprocess();
                   })
        .DefMember("CreateRuntime",
                   [runtime_class, lstate](LuaPmxRuntimeBuilder* lbuilder) -> LuaObject {
                       auto runtime = lbuilder->ptr->CreateRuntime();
                       if (!runtime) {
                           return lstate->CreateNil();
                       }
                       return runtime_class.CreateUserData(lbuilder->engines, runtime);
                   })
        .DefMember("Serialize",
                   [](LuaPmxRuntimeBuilder* lbuilder, const char* output_file, const char* fmt) -> RetCode {
                       return lbuilder->ptr->Serialize(output_file, fmt);
                   });
    l_pmx_module->Set("RuntimeBuilder", lclass);
}

}}}
