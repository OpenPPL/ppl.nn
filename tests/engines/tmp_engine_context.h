#include "ppl/nn/utils/generic_cpu_device.h"
#include "ppl/nn/engines/engine_context.h"

namespace ppl { namespace nn { namespace test {

class TmpEngineContext final : public EngineContext {
public:
    TmpEngineContext(const std::string& name) : name_(name) {}
    Device* GetDevice() override {
        return &device_;
    }

private:
    const std::string name_;
    utils::GenericCpuDevice device_;
};

}}} // namespace ppl::nn::test