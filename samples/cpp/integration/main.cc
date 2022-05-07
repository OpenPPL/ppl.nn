#include "ppl/nn/engines/x86/engine_factory.h"

int main(void) {
    auto x86_engine = ppl::nn::EngineFactory::Create(ppl::nn::EngineOptions());
    delete x86_engine;
    return 0;
}
