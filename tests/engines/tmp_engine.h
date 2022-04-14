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
#include <set>
#include <string>

namespace ppl { namespace nn { namespace test {

class TmpEngine final : public EngineImpl {
public:
    TmpEngine() : supported_types_{"op1", "op2", "op3"} {}
    ppl::common::RetCode Configure(uint32_t, ...) override {
        return ppl::common::RC_UNSUPPORTED;
    }
    EngineContext* CreateEngineContext() override {
        return new TmpEngineContext();
    }
    bool Supports(const ir::Node* node) const override {
        auto& type = node->GetType();
        return (supported_types_.find(type.name) != supported_types_.end());
    }
    ppl::common::RetCode ProcessGraph(const utils::SharedResource&, ir::Graph* graph,
                                      RuntimePartitionInfo* info) override {
        auto topo = graph->topo.get();
        for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
            auto node = it->Get();
            if (supported_types_.find(node->GetType().name) != supported_types_.end()) {
                info->kernels.emplace(node->GetId(), std::unique_ptr<OptKernel>(new TmpOptKernelOne(node)));
            } else {
                return ppl::common::RC_UNSUPPORTED;
            }
        }
        return ppl::common::RC_SUCCESS;
    }
    EngineImpl* Create() override {
        return new TmpEngine();
    }
    const char* GetName() const override {
        return "TmpEngine";
    }
#ifdef PPLNN_ENABLE_PMX_MODEL
    ppl::common::RetCode LoadConstants(const ConstantVisitor&, std::map<edgeid_t, BufferInfo>*) override {
        return ppl::common::RC_SUCCESS;
    }
    OptKernel* CreateOptKernel(const ir::Node* node) const override {
        return new TmpOptKernelOne(node);
    }
    ppl::common::RetCode SerializeData(const pmx::SerializationContext&, utils::DataStream*) const override {
        return ppl::common::RC_UNSUPPORTED;
    }
    ppl::common::RetCode DeserializeData(const void*, uint64_t) override {
        return ppl::common::RC_UNSUPPORTED;
    }
#endif

private:
    utils::GenericCpuDevice device_;
    std::set<std::string> supported_types_;
};

class TmpEngine1 final : public EngineImpl {
public:
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
    ppl::common::RetCode ProcessGraph(const utils::SharedResource&, ir::Graph* graph,
                                      RuntimePartitionInfo* info) override {
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
    EngineImpl* Create() override {
        return new TmpEngine1();
    }
    const char* GetName() const override {
        return "TmpEngine1";
    }
#ifdef PPLNN_ENABLE_PMX_MODEL
    ppl::common::RetCode LoadConstants(const ConstantVisitor&, std::map<edgeid_t, BufferInfo>*) override {
        return ppl::common::RC_SUCCESS;
    }
    OptKernel* CreateOptKernel(const ir::Node* node) const override {
        return new TmpOptKernelOne(node);
    }
    ppl::common::RetCode SerializeData(const pmx::SerializationContext&, utils::DataStream*) const override {
        return ppl::common::RC_UNSUPPORTED;
    }
    ppl::common::RetCode DeserializeData(const void*, uint64_t) override {
        return ppl::common::RC_UNSUPPORTED;
    }
#endif

private:
    utils::GenericCpuDevice device_;
};

class TmpEngine2 final : public EngineImpl {
public:
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
    ppl::common::RetCode ProcessGraph(const utils::SharedResource&, ir::Graph* graph,
                                      RuntimePartitionInfo* info) override {
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
    EngineImpl* Create() override {
        return new TmpEngine2();
    }
    const char* GetName() const override {
        return "TmpEngine2";
    }
#ifdef PPLNN_ENABLE_PMX_MODEL
    ppl::common::RetCode LoadConstants(const ConstantVisitor&, std::map<edgeid_t, BufferInfo>*) override {
        return ppl::common::RC_SUCCESS;
    }
    OptKernel* CreateOptKernel(const ir::Node* node) const override {
        return new TmpOptKernelTwo(node);
    }
    ppl::common::RetCode SerializeData(const pmx::SerializationContext&, utils::DataStream*) const override {
        return ppl::common::RC_UNSUPPORTED;
    }
    ppl::common::RetCode DeserializeData(const void*, uint64_t) override {
        return ppl::common::RC_UNSUPPORTED;
    }
#endif

private:
    utils::GenericCpuDevice device_;
};

class TmpEngine3 final : public EngineImpl {
public:
    ppl::common::RetCode Configure(uint32_t, ...) override {
        return ppl::common::RC_UNSUPPORTED;
    }
    EngineContext* CreateEngineContext() override {
        return new TmpEngineContext();
    }
    bool Supports(const ir::Node* node) const override {
        auto& type = node->GetType();
        return (type.name == "op3");
    }
    ppl::common::RetCode ProcessGraph(const utils::SharedResource&, ir::Graph* graph,
                                      RuntimePartitionInfo* info) override {
        auto topo = graph->topo.get();
        for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
            auto node = it->Get();
            if (node->GetType().name == "op3") {
                info->kernels.emplace(node->GetId(), std::unique_ptr<OptKernel>(new TmpOptKernelTwo(node)));
            } else {
                return ppl::common::RC_UNSUPPORTED;
            }
        }
        return ppl::common::RC_SUCCESS;
    }
    EngineImpl* Create() override {
        return new TmpEngine3();
    }
    const char* GetName() const override {
        return "TmpEngine3";
    }
#ifdef PPLNN_ENABLE_PMX_MODEL
    ppl::common::RetCode LoadConstants(const ConstantVisitor&, std::map<edgeid_t, BufferInfo>*) override {
        return ppl::common::RC_SUCCESS;
    }
    OptKernel* CreateOptKernel(const ir::Node* node) const override {
        return new TmpOptKernelTwo(node);
    }
    ppl::common::RetCode SerializeData(const pmx::SerializationContext&, utils::DataStream*) const override {
        return ppl::common::RC_UNSUPPORTED;
    }
    ppl::common::RetCode DeserializeData(const void*, uint64_t) override {
        return ppl::common::RC_UNSUPPORTED;
    }
#endif

private:
    utils::GenericCpuDevice device_;
};

}}} // namespace ppl::nn::test
