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

#include "tmp_engine_context.h"
#include "ppl/nn/engines/engine_impl.h"
#include "ppl/nn/runtime/runtime_partition_info.h"
#include "tests/engines/tmp_kernel.h"

namespace ppl { namespace nn { namespace test {

class TmpEngineOne final : public EngineImpl {
public:
    TmpEngineOne() : EngineImpl("tmpOne") {}
    ppl::common::RetCode Configure(uint32_t, ...) override {
        return ppl::common::RC_UNSUPPORTED;
    }
    EngineContext* CreateEngineContext() override {
        return new TmpEngineContext();
    }
    bool Supports(const ir::Node* node) const override {
        auto& type = node->GetType();
        return (type.name == "op1");
    }
    ppl::common::RetCode ProcessGraph(utils::SharedResource*, ir::Graph* graph, RuntimePartitionInfo* info) override {
        auto topo = graph->topo.get();
        for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
            auto node = it->Get();
            if (node->GetType().name == "op1") {
                info->kernels.emplace(node->GetId(), std::unique_ptr<OptKernel>(new TmpOptKernelOne(node)));
            } else {
                return ppl::common::RC_UNSUPPORTED;
            }
        }
        return ppl::common::RC_SUCCESS;
    }

private:
    utils::GenericCpuDevice device_;
};

class TmpEngineTwo final : public EngineImpl {
public:
    TmpEngineTwo() : EngineImpl("tmpTwo") {}
    ppl::common::RetCode Configure(uint32_t, ...) override {
        return ppl::common::RC_UNSUPPORTED;
    }
    EngineContext* CreateEngineContext() override {
        return new TmpEngineContext();
    }
    bool Supports(const ir::Node* node) const override {
        auto& type = node->GetType();
        return (type.name == "op2");
    }
    ppl::common::RetCode ProcessGraph(utils::SharedResource*, ir::Graph* graph, RuntimePartitionInfo* info) override {
        auto topo = graph->topo.get();
        for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
            auto node = it->Get();
            if (node->GetType().name == "op2") {
                info->kernels.emplace(node->GetId(), std::unique_ptr<OptKernel>(new TmpOptKernelTwo(node)));
            } else {
                return ppl::common::RC_UNSUPPORTED;
            }
        }
        return ppl::common::RC_SUCCESS;
    }

private:
    utils::GenericCpuDevice device_;
};

}}} // namespace ppl::nn::test
