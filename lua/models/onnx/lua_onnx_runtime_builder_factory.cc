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
#include "ppl/nn/models/onnx/onnx_runtime_builder_factory.h"
#include "../../runtime/lua_runtime_builder.h"
#include "luacpp.h"
using namespace std;
using namespace luacpp;

namespace ppl { namespace nn { namespace lua {

void RegisterOnnxRuntimeBuilderFactory(const shared_ptr<LuaState>& lstate, const shared_ptr<LuaTable>& lmodule) {
    auto builder_class = LuaClass<LuaRuntimeBuilder>(lmodule->Get("RuntimeBuilder"));

    auto lclass = lstate->CreateClass<OnnxRuntimeBuilderFactory>()
        .DefStatic("CreateFromFile", [builder_class, lstate](const char* model_file, const LuaTable& engines) -> LuaObject {
            vector<shared_ptr<Engine>> engine_list;
            engines.ForEach([&engine_list](uint32_t, const LuaObject& value) -> bool {
                engine_list.push_back(LuaUserData(value).Get<LuaEngine>()->ptr);
                return true;
            });

            vector<Engine*> engine_ptrs(engine_list.size());
            for (uint32_t i = 0; i < engine_list.size(); ++i) {
                engine_ptrs[i] = engine_list[i].get();
            }

            auto builder = OnnxRuntimeBuilderFactory::Create(model_file, engine_ptrs.data(), engine_ptrs.size());
            if (!builder) {
                return lstate->CreateNil();
            }

            return builder_class.CreateUserData(engine_list, builder);
        });
    lmodule->Set("OnnxRuntimeBuilderFactory", lclass);
}

}}}
