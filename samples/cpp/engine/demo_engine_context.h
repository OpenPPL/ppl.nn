#ifndef _ST_HPC_PPL_NN_SAMPLES_CPP_ENGINE_DEMO_ENGINE_CONTEXT_H_
#define _ST_HPC_PPL_NN_SAMPLES_CPP_ENGINE_DEMO_ENGINE_CONTEXT_H_

#include "ppl/nn/utils/generic_cpu_device.h"
#include "ppl/nn/engines/engine_context.h"

namespace ppl { namespace nn { namespace demo {

class DemoEngineContext final : public EngineContext {
public:
    DemoEngineContext(const std::string& name) : name_(name) {}
    Device* GetDevice() override {
        return &device_;
    }

private:
    const std::string name_;
    utils::GenericCpuDevice device_;
};

}}} // namespace ppl::nn::demo

#endif
