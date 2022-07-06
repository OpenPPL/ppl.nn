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
#include <memory>
using namespace std;
using namespace luacpp;

namespace ppl { namespace nn { namespace lua {

void RegisterVersion(const shared_ptr<LuaState>& lstate, const shared_ptr<LuaTable>& lmodule) {
    lmodule->SetInteger("PPLNN_VERSION_MAJOR", PPLNN_VERSION_MAJOR);
    lmodule->SetInteger("PPLNN_VERSION_MINOR", PPLNN_VERSION_MINOR);
    lmodule->SetInteger("PPLNN_VERSION_PATCH", PPLNN_VERSION_PATCH);
    lmodule->SetString("PPLNN_COMMIT_STR", PPLNN_COMMIT_STR);
}

}}} // namespace ppl::nn::lua
