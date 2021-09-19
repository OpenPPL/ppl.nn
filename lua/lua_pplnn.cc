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

#include "ppl/nn/common/common.h"
#include "luacpp.h"
#include <memory>
using namespace std;
using namespace luacpp;

namespace ppl { namespace nn { namespace lua {

#ifdef PPLNN_USE_X86
void RegisterX86EngineOptions(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterX86Engine(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterX86EngineFactory(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
#endif
#ifdef PPLNN_USE_CUDA
void RegisterCudaEngineOptions(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterCudaEngine(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterCudaEngineFactory(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
#endif

void RegisterGetVersionString(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterTensorShape(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterTensor(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterRuntime(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterRuntimeBuilder(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterOnnxRuntimeBuilderFactory(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);

}}}

using namespace ppl::nn::lua;

extern "C" {

/* require("luappl.nn") */
int PPLNN_PUBLIC luaopen_luappl_nn(lua_State* l) {
    // may be used by module functions outside this function scope
    auto lstate = make_shared<LuaState>(l, false);
    auto lmodule = make_shared<LuaTable>(lstate->CreateTable());

    // NOTE register classes in order

#ifdef PPLNN_USE_CUDA
    RegisterCudaEngineOptions(lstate, lmodule);
    RegisterCudaEngine(lstate, lmodule);
    RegisterCudaEngineFactory(lstate, lmodule);
#endif
#ifdef PPLNN_USE_X86
    RegisterX86EngineOptions(lstate, lmodule);
    RegisterX86Engine(lstate, lmodule);
    RegisterX86EngineFactory(lstate, lmodule);
#endif

    RegisterGetVersionString(lstate, lmodule);
    RegisterTensorShape(lstate, lmodule);
    RegisterTensor(lstate, lmodule);
    RegisterRuntime(lstate, lmodule);
    RegisterRuntimeBuilder(lstate, lmodule);
    RegisterOnnxRuntimeBuilderFactory(lstate, lmodule);

    lmodule->PushSelf();
    return 1;
}

}
