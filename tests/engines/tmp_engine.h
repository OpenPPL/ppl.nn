#include "ppl/nn/engines/engine_impl.h"
#include "ppl/nn/utils/generic_cpu_device.h"
#include "ppl/nn/runtime/runtime_partition_info.h"
#include "tests/engines/tmp_engine_context.h"
#include "tests/engines/tmp_kernel.h"

namespace ppl { namespace nn { namespace test {

class TmpEngineOne final : public EngineImpl {
public:
    TmpEngineOne() : EngineImpl("tmpOne") {}
    ppl::common::RetCode Configure(uint32_t, ...) override {
        return ppl::common::RC_UNSUPPORTED;
    }
    EngineContext* CreateEngineContext(const std::string&, const EngineContextOptions&) override {
        return new TmpEngineContext(GetName());
    }
    bool CanRunOp(const ir::Node* node) const override {
        auto& type = node->GetType();
        return (type.name == "op1" || type.name == "op2");
    }
    ppl::common::RetCode ProcessGraph(utils::SharedResource*, ir::Graph* graph, RuntimePartitionInfo* info) override {
        auto topo = graph->topo.get();
        for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
            auto node = it->Get();
            if (node->GetType().name == "op1") {
                info->kernels.emplace(node->GetId(), std::unique_ptr<OptKernel>(new TmpOptKernelOne(node)));
            } else {
                info->kernels.emplace(node->GetId(), std::unique_ptr<OptKernel>(new TmpOptKernelTwo(node)));
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
    EngineContext* CreateEngineContext(const std::string&, const EngineContextOptions&) override {
        return new TmpEngineContext(GetName());
    }
    bool CanRunOp(const ir::Node* node) const override {
        auto& type = node->GetType();
        return (type.name == "op3" || type.name == "op4");
    }
    ppl::common::RetCode ProcessGraph(utils::SharedResource*, ir::Graph* graph, RuntimePartitionInfo* info) override {
        auto topo = graph->topo.get();
        for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
            auto node = it->Get();
            if (node->GetType().name == "op3") {
                info->kernels.emplace(node->GetId(), std::unique_ptr<OptKernel>(new TmpOptKernelOne(node)));
            } else {
                info->kernels.emplace(node->GetId(), std::unique_ptr<OptKernel>(new TmpOptKernelTwo(node)));
            }
        }
        return ppl::common::RC_SUCCESS;
    }

private:
    utils::GenericCpuDevice device_;
};

}}} // namespace ppl::nn::test
