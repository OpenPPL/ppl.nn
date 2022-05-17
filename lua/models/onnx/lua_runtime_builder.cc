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
#include "lua_runtime_builder.h"
#include "lua_runtime_builder_resources.h"
#include "luacpp/luacpp.h"
using namespace std;
using namespace luacpp;
using namespace ppl::common;
using namespace ppl::nn::onnx;

namespace ppl { namespace nn { namespace lua { namespace onnx {

void RegisterRuntimeBuilder(const shared_ptr<LuaState>& lstate, const shared_ptr<LuaTable>& l_onnx_module,
                            const shared_ptr<LuaTable>& lmodule) {
    auto runtime_class = lmodule->GetClass<LuaRuntime>("Runtime");
    auto lclass = lstate->CreateClass<LuaRuntimeBuilder>()
        .DefMember("LoadModelFromFile",
                   [](LuaRuntimeBuilder* lbuilder, const char* model_file) -> RetCode {
                       return lbuilder->ptr->LoadModel(model_file);
                   })
        .DefMember("SetResources",
                   [](LuaRuntimeBuilder* lbuilder, const LuaRuntimeBuilderResources* resources) -> RetCode {
                       lbuilder->engines = resources->engines;

                       vector<Engine*> engine_ptrs(resources->engines.size());
                       for (uint32_t i = 0; i < resources->engines.size(); ++i) {
                           engine_ptrs[i] = resources->engines[i].get();
                       }
                       RuntimeBuilder::Resources r;
                       r.engines = engine_ptrs.data();
                       r.engine_num = engine_ptrs.size();

                       return lbuilder->ptr->SetResources(r);
                   })
        .DefMember("Preprocess",
                   [](LuaRuntimeBuilder* lbuilder) -> RetCode {
                       return lbuilder->ptr->Preprocess();
                   })
        .DefMember("CreateRuntime",
                   [runtime_class, lstate](LuaRuntimeBuilder* lbuilder) -> LuaObject {
                       auto runtime = lbuilder->ptr->CreateRuntime();
                       if (!runtime) {
                           return lstate->CreateNil();
                       }
                       return runtime_class.CreateInstance(lbuilder->engines, runtime);
                   })
        .DefMember("Serialize",
                   [](LuaRuntimeBuilder* lbuilder, const char* output_file, const char* fmt) -> RetCode {
                       return lbuilder->ptr->Serialize(output_file, fmt);
                   });
    l_onnx_module->Set("RuntimeBuilder", lclass);
}

}}}} // namespace ppl::nn::lua::onnx
