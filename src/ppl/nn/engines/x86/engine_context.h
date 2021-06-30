#ifndef _ST_HPC_PPL_NN_ENGINES_X86_ENGINE_CONTEXT_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_ENGINE_CONTEXT_H_

#include "ppl/nn/engines/x86/x86_device.h"
#include "ppl/nn/engines/x86/kernel.h"
#include "ppl/nn/engines/x86/runtime_x86_device.h"
#include "ppl/nn/engines/engine_context.h"
#include "ppl/nn/engines/engine_context_options.h"

namespace ppl { namespace nn { namespace x86 {

#define X86_DEFAULT_ALIGNMENT 64u

class X86EngineContext final : public EngineContext {
public:
    X86EngineContext(const std::string& name, ppl::common::isa_t isa, const EngineContextOptions& options)
        : name_(name), device_(X86_DEFAULT_ALIGNMENT, isa, options.mm_policy) {}
    Device* GetDevice() override {
        return &device_;
    }

private:
    const std::string name_;
    RuntimeX86Device device_;
};

}}} // namespace ppl::nn::x86

#endif
