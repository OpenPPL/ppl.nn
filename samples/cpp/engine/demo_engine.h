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

#ifndef _ST_HPC_PPL_NN_SAMPLES_CPP_ENGINE_DEMO_ENGINE_H_
#define _ST_HPC_PPL_NN_SAMPLES_CPP_ENGINE_DEMO_ENGINE_H_

#include "ppl/nn/engines/engine_impl.h"
#include "ppl/nn/utils/generic_cpu_device.h"

namespace ppl { namespace nn { namespace demo {

class DemoEngine final : public EngineImpl {
public:
    DemoEngine() : EngineImpl("demo") {}
    ppl::common::RetCode Configure(uint32_t, ...) override {
        return ppl::common::RC_UNSUPPORTED;
    }
    EngineContext* CreateEngineContext() override;
    bool Supports(const ir::Node*) const override {
        return true;
    }
    ppl::common::RetCode ProcessGraph(utils::SharedResource*, ir::Graph*, RuntimePartitionInfo*) override;

private:
    utils::GenericCpuDevice device_;
};

}}} // namespace ppl::nn::demo

#endif
