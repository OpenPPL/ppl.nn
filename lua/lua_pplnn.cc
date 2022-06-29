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

#include "luacpp/luacpp.h"
using namespace luacpp;

#include "ppl/nn/common/common.h"
#include "ppl/nn/common/logger.h"
#include "ppl/common/retcode.h"
using namespace ppl::common;

#include <memory>
#include <map>
using namespace std;

namespace ppl { namespace nn { namespace lua {

#ifdef PPLNN_USE_X86
namespace x86 {
void RegisterBuiltinOpImpls();
void RegisterEngineOptions(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterEngine(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterEngineFactory(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
} // namespace x86
#endif

#ifdef PPLNN_USE_CUDA
namespace cuda {
void RegisterBuiltinOpImpls();
void RegisterEngineOptions(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterEngine(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterEngineFactory(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
} // namespace cuda
#endif

#ifdef PPLNN_USE_RISCV
namespace riscv {
void RegisterBuiltinOpImpls();
void RegisterEngineOptions(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterEngine(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterEngineFactory(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
} // namespace riscv
#endif

#ifdef PPLNN_USE_ARM
namespace arm {
void RegisterBuiltinOpImpls();
void RegisterEngineOptions(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterEngine(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterEngineFactory(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
} // namespace arm
#endif

void RegisterVersion(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterEngine(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterTensorShape(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterTensor(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
void RegisterRuntime(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);

#ifdef PPLNN_ENABLE_ONNX_MODEL
namespace onnx {
void RegisterRuntimeBuilderResources(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&,
                                     const shared_ptr<LuaTable>&);
void RegisterRuntimeBuilder(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&, const shared_ptr<LuaTable>&);
void RegisterRuntimeBuilderFactory(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
} // namespace onnx
#endif

#ifdef PPLNN_ENABLE_PMX_MODEL
namespace pmx {
void RegisterRuntimeBuilderResources(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&,
                                     const shared_ptr<LuaTable>&);
void RegisterRuntimeBuilder(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&, const shared_ptr<LuaTable>&);
void RegisterRuntimeBuilderFactory(const shared_ptr<LuaState>&, const shared_ptr<LuaTable>&);
} // namespace pmx
#endif

// this function's implementation is in PPLNN_LUA_API_EXTERNAL_SOURCES
void LoadResources(const shared_ptr<LuaState>&, const map<string, shared_ptr<LuaTable>>&);

/* require("luappl.nn") */
extern "C" int PPLNN_PUBLIC luaopen_luappl_nn(lua_State* l) {
    // may be used by module functions outside this function scope
    auto lstate = make_shared<LuaState>(l, false);
    auto lmodule = make_shared<LuaTable>(lstate->CreateTable());

    map<string, shared_ptr<LuaTable>> name2module;

    // root module's name is an empty string
    name2module.insert(make_pair("", lmodule));

    // NOTE register classes in order

#ifdef PPLNN_USE_CUDA
    auto l_cuda_module = make_shared<LuaTable>(lstate->CreateTable());
    cuda::RegisterBuiltinOpImpls();
    cuda::RegisterEngineOptions(lstate, l_cuda_module);
    cuda::RegisterEngine(lstate, l_cuda_module);
    cuda::RegisterEngineFactory(lstate, l_cuda_module);
    lmodule->Set("cuda", *l_cuda_module);
    name2module.insert(make_pair("cuda", l_cuda_module));
#endif

#ifdef PPLNN_USE_X86
    auto l_x86_module = make_shared<LuaTable>(lstate->CreateTable());
    x86::RegisterBuiltinOpImpls();
    x86::RegisterEngineOptions(lstate, l_x86_module);
    x86::RegisterEngine(lstate, l_x86_module);
    x86::RegisterEngineFactory(lstate, l_x86_module);
    lmodule->Set("x86", *l_x86_module);
    name2module.insert(make_pair("x86", l_x86_module));
#endif

#ifdef PPLNN_USE_RISCV
    auto l_riscv_module = make_shared<LuaTable>(lstate->CreateTable());
    riscv::RegisterBuiltinOpImpls();
    riscv::RegisterEngineOptions(lstate, l_riscv_module);
    riscv::RegisterEngine(lstate, l_riscv_module);
    riscv::RegisterEngineFactory(lstate, l_riscv_module);
    lmodule->Set("riscv", *l_riscv_module);
    name2module.insert(make_pair("riscv", l_riscv_module));
#endif

#ifdef PPLNN_USE_ARM
    auto l_arm_module = make_shared<LuaTable>(lstate->CreateTable());
    arm::RegisterBuiltinOpImpls();
    arm::RegisterEngineOptions(lstate, l_arm_module);
    arm::RegisterEngine(lstate, l_arm_module);
    arm::RegisterEngineFactory(lstate, l_arm_module);
    lmodule->Set("arm", *l_arm_module);
    name2module.insert(make_pair("arm", l_arm_module));
#endif

    RegisterVersion(lstate, lmodule);
    RegisterEngine(lstate, lmodule);
    RegisterTensorShape(lstate, lmodule);
    RegisterTensor(lstate, lmodule);
    RegisterRuntime(lstate, lmodule);

#ifdef PPLNN_ENABLE_ONNX_MODEL
    auto l_onnx_module = make_shared<LuaTable>(lstate->CreateTable());
    onnx::RegisterRuntimeBuilderResources(lstate, l_onnx_module, lmodule);
    onnx::RegisterRuntimeBuilder(lstate, l_onnx_module, lmodule);
    onnx::RegisterRuntimeBuilderFactory(lstate, l_onnx_module);
    lmodule->Set("onnx", *l_onnx_module);
    name2module.insert(make_pair("onnx", l_onnx_module));
#endif

#ifdef PPLNN_ENABLE_PMX_MODEL
    auto l_pmx_module = make_shared<LuaTable>(lstate->CreateTable());
    pmx::RegisterRuntimeBuilderResources(lstate, l_pmx_module, lmodule);
    pmx::RegisterRuntimeBuilder(lstate, l_pmx_module, lmodule);
    pmx::RegisterRuntimeBuilderFactory(lstate, l_pmx_module);
    lmodule->Set("pmx", *l_pmx_module);
    name2module.insert(make_pair("pmx", l_pmx_module));
#endif

    LoadResources(lstate, name2module);

    lstate->Push(*lmodule);
    return 1;
}

}}} // namespace ppl::nn::lua
