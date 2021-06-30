#include "ppl/nn/engines/cuda/engine_context.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/default_cuda_device.h"
#include "ppl/nn/engines/cuda/buffered_cuda_device.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode CudaEngineContext::Init(const EngineContextOptions& options) {
    if (options.mm_policy != MM_LESS_MEMORY) {
        LOG(WARNING) << "unsupported mm policy[" << options.mm_policy << "]. CudaEngine supports MM_LESS_MEMORY only."
                     << " mm policy will be MM_LESS_MEMORY.";
    }

    // TODO implement other options
    auto status = device_.Init(MM_LESS_MEMORY);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init BufferedCudaDevice failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
