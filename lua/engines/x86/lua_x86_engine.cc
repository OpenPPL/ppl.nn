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

#ifdef PPLNN_USE_X86

#include "../lua_engine.h"
#include "luacpp.h"
#include "ppl/nn/engines/x86/x86_options.h"
#include "ppl/nn/common/logger.h"
#include <map>
using namespace std;
using namespace luacpp;
using namespace ppl::common;

namespace ppl { namespace nn { namespace lua {

static RetCode GenericSetOption(Engine* engine, uint32_t option, const LuaObject&) {
    return engine->Configure(option);
}

typedef RetCode (*ConfigFunc)(Engine*, uint32_t option, const LuaObject& args);

static const map<uint32_t, ConfigFunc> g_opt2func = {
    {X86_CONF_DISABLE_AVX512, GenericSetOption},
    {X86_CONF_DISABLE_AVX_FMA3, GenericSetOption},
};

void RegisterX86Engine(const shared_ptr<LuaState>& lstate, const shared_ptr<LuaTable>& lmodule) {
    auto lclass = lstate->CreateClass<LuaEngine>()
        .DefMember("GetName", [](const LuaEngine* engine) -> const char* {
            return engine->ptr->GetName();
        })
        .DefMember("Configure", [](LuaEngine* lengine, uint32_t option, const LuaObject& args) -> RetCode {
            auto it = g_opt2func.find(option);
            if (it == g_opt2func.end()) {
                LOG(ERROR) << "unsupported option: " << option;
                return RC_UNSUPPORTED;
            }
            return it->second(lengine->ptr.get(), option, args);
        });
    lmodule->Set("X86Engine", lclass);

    lmodule->SetInteger("X86_CONF_DISABLE_AVX512", X86_CONF_DISABLE_AVX512);
    lmodule->SetInteger("X86_CONF_DISABLE_AVX_FMA3", X86_CONF_DISABLE_AVX_FMA3);
}

}}}

#endif
