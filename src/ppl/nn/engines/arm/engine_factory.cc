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

#include "ppl/nn/engines/arm/engine_factory.h"
#include "ppl/nn/engines/arm/engine.h"
#include "ppl/common/sys.h"
#include "ppl/nn/common/logger.h"

namespace ppl { namespace nn { namespace arm {

Engine* EngineFactory::Create(const EngineOptions& options) {
    auto engine = new arm::ArmEngine();
    if (engine) {
        auto status = engine->Init(options);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "Init arm engine failed: " << ppl::common::GetRetCodeStr(status);
            delete engine;
            return nullptr;
        }
    }
    return engine;
}

}}} // namespace ppl::nn::arm
