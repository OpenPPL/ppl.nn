#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_ENGINE_CONTEXT_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_ENGINE_CONTEXT_H_

#include "ppl/nn/engines/cuda/buffered_cuda_device.h"
#include "ppl/nn/engines/engine_context.h"
#include "ppl/nn/engines/engine_context_options.h"

namespace ppl { namespace nn { namespace cuda {

class CudaEngineContext final : public EngineContext {
public:
    CudaEngineContext(const std::string& name) : name_(name) {}
    ppl::common::RetCode Init(const EngineContextOptions& options);
    Device* GetDevice() override {
        return &device_;
    }

private:
    const std::string name_;
    BufferedCudaDevice device_;
};

}}} // namespace ppl::nn::cuda

#endif
