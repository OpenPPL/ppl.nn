#include "ppl/nn/engines/cuda/engine.h"
#include "ppl/nn/engines/cuda/engine_factory.h"
#include "ppl/nn/common/logger.h"
using namespace ppl::common;

namespace ppl { namespace nn {

Engine* CudaEngineFactory::Create() {
    auto engine = new cuda::CudaEngine();
    if (engine) {
        auto status = engine->Init();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "init cuda engine failed: " << GetRetCodeStr(status);
            delete engine;
            return nullptr;
        }
    }
    return engine;
}

}} // namespace ppl::nn
