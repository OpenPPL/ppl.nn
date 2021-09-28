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
#include "luacpp.h"
#include "lua_tensor.h"
using namespace std;
using namespace luacpp;
using namespace ppl::common;

namespace ppl { namespace nn { namespace lua {

void RegisterRuntime(const shared_ptr<LuaState>& lstate, const shared_ptr<LuaTable>& lmodule) {
    auto tensor_class = LuaClass<LuaTensor>(lmodule->Get("Tensor"));
    auto lclass = lstate->CreateClass<LuaRuntime>()
        .DefMember("GetInputCount", [](const LuaRuntime* lruntime) -> uint32_t {
            return lruntime->ptr->GetInputCount();
        })
        .DefMember("GetInputTensor", [tensor_class, lstate](const LuaRuntime* lruntime, uint32_t idx) -> LuaObject {
            auto tensor = lruntime->ptr->GetInputTensor(idx);
            if (!tensor) {
                return lstate->CreateNil();
            }
            return tensor_class.CreateUserData(tensor);
        })
        .DefMember("Run", [](LuaRuntime* lruntime) -> RetCode {
            return lruntime->ptr->Run();
        })
        .DefMember("Sync", [](LuaRuntime* lruntime) -> RetCode {
            return lruntime->ptr->Sync();
        })
        .DefMember("GetOutputCount", [](const LuaRuntime* lruntime) -> uint32_t {
            return lruntime->ptr->GetOutputCount();
        })
        .DefMember("GetOutputTensor", [tensor_class, lstate](const LuaRuntime* lruntime, uint32_t idx) -> LuaObject {
            auto tensor = lruntime->ptr->GetOutputTensor(idx);
            if (!tensor) {
                return lstate->CreateNil();
            }
            return tensor_class.CreateUserData(tensor);
        });

    lmodule->Set("Runtime", lclass);
}

}}}
