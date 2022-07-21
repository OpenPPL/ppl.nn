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

#include "ppl/nn/engines/arm/engine_context.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

RetCode ArmEngineContext::Init(isa_t isa, uint32_t mm_policy) {
    if (mm_policy == MM_PLAIN) {
        device_ = make_shared<ArmDevice>(ARM_DEFAULT_ALIGNMENT, isa);
    } else {
        auto dev = make_shared<RuntimeArmDevice>(ARM_DEFAULT_ALIGNMENT, isa);
        auto rc = dev->Init(mm_policy);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "init RuntimeArmDevice failed: " << GetRetCodeStr(rc);
            return rc;
        }
        device_ = dev;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::arm
