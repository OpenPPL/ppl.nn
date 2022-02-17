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

#include <memory>
using namespace std;

#include "luacpp/luacpp.h"
using namespace luacpp;

#include "ppl/nn/common/common.h"
#include "ppl/nn/common/logger.h"
#include "ppl/common/retcode.h"
using namespace ppl::common;

#include "lua_type_creator_manager.h"

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

#ifdef PPLNN_USE_RISCV
void RegisterRiscvEngineOptions(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterRiscvEngine(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterRiscvEngineFactory(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
#endif

#ifdef PPLNN_USE_ARM
void RegisterArmEngineOptions(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterArmEngine(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterArmEngineFactory(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
#endif

void RegisterGetVersionString(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterTensorShape(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterTensor(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterRuntime(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);

#ifdef PPLNN_ENABLE_ONNX_MODEL
void RegisterOnnxRuntimeBuilder(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterOnnxRuntimeBuilderFactory(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
#endif

#ifdef PPLNN_ENABLE_PMX_MODEL
void RegisterPmxRuntimeBuilder(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterPmxRuntimeBuilderFactory(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
#endif

/* require("luappl.nn") */
extern "C" int PPLNN_PUBLIC luaopen_luappl_nn(lua_State* l) {
    // may be used by module functions outside this function scope
    auto lstate = make_shared<LuaState>(l, false);
    auto lmodule = make_shared<LuaTable>(lstate->CreateTable());

    auto mgr = LuaTypeCreatorManager::Instance();
    for (uint32_t i = 0; i < mgr->GetCreatorCount(); ++i) {
        auto creator = mgr->GetCreator(i);
        auto status = creator->Register(lstate, lmodule);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "register lua type failed.";
            lstate->PushNil();
            return 1;
        }
    }

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

#ifdef PPLNN_USE_RISCV
    RegisterRiscvEngineOptions(lstate, lmodule);
    RegisterRiscvEngine(lstate, lmodule);
    RegisterRiscvEngineFactory(lstate, lmodule);
#endif

#ifdef PPLNN_USE_ARM
    RegisterArmEngineOptions(lstate, lmodule);
    RegisterArmEngine(lstate, lmodule);
    RegisterArmEngineFactory(lstate, lmodule);
#endif

    RegisterGetVersionString(lstate, lmodule);
    RegisterTensorShape(lstate, lmodule);
    RegisterTensor(lstate, lmodule);
    RegisterRuntime(lstate, lmodule);

#ifdef PPLNN_ENABLE_ONNX_MODEL
    RegisterOnnxRuntimeBuilder(lstate, lmodule);
    RegisterOnnxRuntimeBuilderFactory(lstate, lmodule);
#endif

#ifdef PPLNN_ENABLE_PMX_MODEL
    RegisterPmxRuntimeBuilder(lstate, lmodule);
    RegisterPmxRuntimeBuilderFactory(lstate, lmodule);
#endif

    lstate->Push(*lmodule);
    return 1;
}

}}} // namespace ppl::nn::lua
