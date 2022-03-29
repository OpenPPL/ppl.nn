#include "ppl/nn/engines/x86/engine_factory.h"

int main(void) {
    auto x86_engine = ppl::nn::X86EngineFactory::Create(ppl::nn::X86EngineOptions());
    delete x86_engine;
    return 0;
}
