#include "ppl/nn/engines/x86/engine_factory.h"
#include "ppl/nn/engines/x86/engine.h"

namespace ppl { namespace nn {

Engine* X86EngineFactory::Create() {
    return new x86::X86Engine();
}

}} // namespace ppl::nn
