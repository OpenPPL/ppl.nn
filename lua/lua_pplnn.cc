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
void RegisterX86BuiltinOpImpls();
void RegisterX86EngineOptions(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterX86Engine(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterX86EngineFactory(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
#endif

#ifdef PPLNN_USE_CUDA
void RegisterCudaBuiltinOpImpls();
void RegisterCudaEngineOptions(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterCudaEngine(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterCudaEngineFactory(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
#endif

#ifdef PPLNN_USE_RISCV
void RegisterRiscvBuiltinOpImpls();
void RegisterRiscvEngineOptions(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterRiscvEngine(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterRiscvEngineFactory(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
#endif

#ifdef PPLNN_USE_ARM
void RegisterArmBuiltinOpImpls();
void RegisterArmEngineOptions(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterArmEngine(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterArmEngineFactory(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
#endif

void RegisterVersion(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterTensorShape(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterTensor(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterRuntime(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);

#ifdef PPLNN_ENABLE_ONNX_MODEL
void RegisterOnnxRuntimeBuilder(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&, const shared_ptr<LuaTable>&);
void RegisterOnnxRuntimeBuilderFactory(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
#endif

#ifdef PPLNN_ENABLE_PMX_MODEL
void RegisterPmxRuntimeBuilder(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&, const shared_ptr<LuaTable>&);
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
    auto l_cuda_module = make_shared<LuaTable>(lstate->CreateTable());
    RegisterCudaBuiltinOpImpls();
    RegisterCudaEngineOptions(lstate, l_cuda_module);
    RegisterCudaEngine(lstate, l_cuda_module);
    RegisterCudaEngineFactory(lstate, l_cuda_module);
    lmodule->Set("cuda", *l_cuda_module);
#endif

#ifdef PPLNN_USE_X86
    auto l_x86_module = make_shared<LuaTable>(lstate->CreateTable());
    RegisterX86BuiltinOpImpls();
    RegisterX86EngineOptions(lstate, l_x86_module);
    RegisterX86Engine(lstate, l_x86_module);
    RegisterX86EngineFactory(lstate, l_x86_module);
    lmodule->Set("x86", *l_x86_module);
#endif

#ifdef PPLNN_USE_RISCV
    auto l_riscv_module = make_shared<LuaTable>(lstate->CreateTable());
    RegisterRiscvBuiltinOpImpls();
    RegisterRiscvEngineOptions(lstate, l_riscv_module);
    RegisterRiscvEngine(lstate, l_riscv_module);
    RegisterRiscvEngineFactory(lstate, l_riscv_module);
    lmodule->Set("riscv", *l_riscv_module);
#endif

#ifdef PPLNN_USE_ARM
    auto l_arm_module = make_shared<LuaTable>(lstate->CreateTable());
    RegisterArmBuiltinOpImpls();
    RegisterArmEngineOptions(lstate, l_arm_module);
    RegisterArmEngine(lstate, l_arm_module);
    RegisterArmEngineFactory(lstate, l_arm_module);
    lmodule->Set("arm", *l_arm_module);
#endif

    RegisterVersion(lstate, lmodule);
    RegisterTensorShape(lstate, lmodule);
    RegisterTensor(lstate, lmodule);
    RegisterRuntime(lstate, lmodule);

#ifdef PPLNN_ENABLE_ONNX_MODEL
    auto l_onnx_module = make_shared<LuaTable>(lstate->CreateTable());
    RegisterOnnxRuntimeBuilder(lstate, lmodule, l_onnx_module);
    RegisterOnnxRuntimeBuilderFactory(lstate, l_onnx_module);
    lmodule->Set("onnx", *l_onnx_module);
#endif

#ifdef PPLNN_ENABLE_PMX_MODEL
    auto l_pmx_module = make_shared<LuaTable>(lstate->CreateTable());
    RegisterPmxRuntimeBuilder(lstate, lmodule, l_pmx_module);
    RegisterPmxRuntimeBuilderFactory(lstate, l_pmx_module);
    lmodule->Set("pmx", *l_pmx_module);
#endif

    lstate->Push(*lmodule);
    return 1;
}

}}} // namespace ppl::nn::lua
