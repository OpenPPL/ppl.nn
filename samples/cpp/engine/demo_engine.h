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
    EngineContext* CreateEngineContext(const std::string& graph_name, const EngineContextOptions&) override;
    bool CanRunOp(const ir::Node*) const override {
        return true;
    }
    ppl::common::RetCode ProcessGraph(utils::SharedResource*, ir::Graph*, RuntimePartitionInfo*) override;

private:
    utils::GenericCpuDevice device_;
};

}}} // namespace ppl::nn::demo

#endif
