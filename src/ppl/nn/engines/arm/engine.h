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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_ENGINE_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_ENGINE_H_

#include "ppl/nn/engines/engine_impl.h"
#include "ppl/nn/engines/arm/arm_device.h"
#include "ppl/common/arm/sysinfo.h"
#include "ppl/nn/engines/arm/arm_engine_options.h"

namespace ppl { namespace nn { namespace arm {

class ArmEngine final : public EngineImpl {
public:
    ppl::common::RetCode Init(const ArmEngineOptions&);
    ArmEngine() : EngineImpl("arm"), device_(ARM_DEFAULT_ALIGNMENT, ppl::common::GetCpuISA()){};
    ppl::common::RetCode BindNumaNode(int32_t numa_node_id) const;
    ppl::common::RetCode Configure(uint32_t, ...) override;
    EngineContext* CreateEngineContext() override;
    bool Supports(const ir::Node* node) const override;
    ppl::common::RetCode ProcessGraph(utils::SharedResource*, ir::Graph*, RuntimePartitionInfo*) override;

private:
    ppl::common::RetCode DoOptimize(ir::Graph*, utils::SharedResource*, RuntimePartitionInfo*);

    typedef ppl::common::RetCode (*ConfHandlerFunc)(ArmEngine*, va_list);
    static ConfHandlerFunc conf_handlers_[ARM_CONF_MAX];

private:
    ArmDevice device_;
    ArmEngineOptions options_;
};

}}} // namespace ppl::nn::arm

#endif
